"""Общие функции для ноутбуков ВКР: чтение данных, предобработка, сегментация, признаки, оценка моделей.

Каналы во всех функциях идут в порядке (lyso, mito, nuc) = (LysoTracker Red, MitoTracker Green, Hoechst).
"""
import os, re, itertools
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import exposure, feature, filters, measure, morphology, segmentation, transform
from skimage.feature import graycomatrix, graycoprops

TARGET_PX = 0.18  # мкм на пиксель после приведения масштаба (наибольший размер пикселя в наборе)
PAT = re.compile(r'^(Acom|Mus)_(Young|Old)_(Control|control|5Gy|10Gy)_(\d)_R(\d)\.tif$')
DOSE = {'Control': 0, 'control': 0, '5Gy': 5, '10Gy': 10}


def _rm_small(mask, size):
    """Удаляет объекты площадью <= size пикселей при любой версии scikit-image
    (в 0.26 параметр min_size заменен на max_size с другой границей)."""
    import inspect
    if 'max_size' in inspect.signature(morphology.remove_small_objects).parameters:
        return morphology.remove_small_objects(mask, max_size=size)
    return morphology.remove_small_objects(mask, min_size=size + 1)


# ---------------------------------------------------------------- данные
def list_dataset(root):
    """Таблица всех изображений с метаданными из имени файла."""
    rows = []
    for group in sorted(os.listdir(root)):
        gdir = os.path.join(root, group)
        if not os.path.isdir(gdir):
            continue
        for fn in sorted(os.listdir(gdir)):
            m = PAT.match(fn)
            if not m:
                continue
            sp, age, dose, cult, rep = m.groups()
            rows.append(dict(file=fn, path=os.path.join(gdir, fn), species=sp, age=age, dose=DOSE[dose],
                             culture_in_type=int(cult), replicate=int(rep)))
    df = pd.DataFrame(rows)
    df['type'] = df.species + '_' + df.age
    df['culture'] = df.type + '_c' + df.culture_in_type.astype(str)  # уникальный идентификатор культуры (животного)
    df['irradiated'] = (df.dose > 0).astype(int)
    return df.reset_index(drop=True)


def _lut_color(lut):
    return 'RGB'[int(np.argmax([lut[c, -1] for c in range(3)]))]


def read_image(path, resample=True):
    """Возвращает (arr[3,H,W] float32 в порядке lyso, mito, nuc; размер пикселя в мкм; исходный порядок LUT).
    Каналы определяются по LUT, потому что в одном файле порядок каналов другой."""
    with tifffile.TiffFile(path) as t:
        arr = t.series[0].asarray().astype(np.float32)
        luts = ''.join(_lut_color(l) for l in t.imagej_metadata['LUTs'])
        xr = t.pages[0].tags['XResolution'].value
    px = xr[1] / xr[0]
    arr = arr[[luts.index(c) for c in 'RGB']]
    if resample:
        s = px / TARGET_PX
        arr = np.stack([transform.rescale(a, s, anti_aliasing=True, preserve_range=True) for a in arr]).astype(np.float32)
        px = TARGET_PX
    return arr, px, luts


def norm01(a, lo=0.5, hi=99.8, mask=None):
    """Нормировка интенсивности внутри изображения (или маски) по перцентилям — только для отображения и нейросетей."""
    v = a[mask] if mask is not None and mask.any() else a
    l, h = np.percentile(v, lo), np.percentile(v, hi)
    return np.clip((a - l) / max(h - l, 1e-6), 0, 1).astype(np.float32)


def composite(arr):
    return np.stack([norm01(arr[0]), norm01(arr[1]), norm01(arr[2])], -1)


def qc_metrics(path):
    """Показатели контроля качества по исходному файлу (без приведения масштаба)."""
    with tifffile.TiffFile(path) as t:
        raw = t.series[0].asarray()
        luts = ''.join(_lut_color(l) for l in t.imagej_metadata['LUTs'])
        xr = t.pages[0].tags['XResolution'].value
    out = dict(px_um=xr[1] / xr[0], luts=luts, shape='x'.join(map(str, raw.shape)))
    for name, c in zip(['lyso', 'mito', 'nuc'], 'RGB'):
        ch = raw[luts.index(c)]
        out[f'{name}_sat_pct'] = float((ch == 255).mean() * 100)
        out[f'{name}_zero_pct'] = float((ch == 0).mean() * 100)
        out[f'{name}_focus'] = focus_measure(ch)
    return out


def focus_measure(ch):
    """Резкость: дисперсия лапласиана после нормировки по перцентилям сигнала (пиксели > 0).
    Нормировка нужна, чтобы мера не зависела от яркости окрашивания и настроек съемки."""
    ch = ch.astype(np.float32)
    sig = ch > 0
    if sig.sum() < 100:
        return np.nan
    z = norm01(ch, 1, 99.5, mask=sig)
    lap = ndi.laplace(ndi.gaussian_filter(z, 1))
    return float(lap[ndi.binary_erosion(sig, iterations=2)].var() * 1e3)


# ---------------------------------------------------------------- сегментация
def seg_cell(arr, px):
    """Маска клеток: сумма сглаженных каналов митохондрий и лизосом, порог треугольника."""
    cs = ndi.gaussian_filter(norm01(arr[0]) + norm01(arr[1]), 4)
    cm = cs > filters.threshold_triangle(cs)
    cm = morphology.closing(cm, morphology.disk(5))
    cm = ndi.binary_fill_holes(cm)
    return _rm_small(cm, int(100 / px ** 2))


def seg_nuclei_otsu(nuc, px, sigma_um=1.0, close_um=2.0, h_um=2.0):
    """Ядра (классический вариант): сглаживание, порог Оцу, закрытие и заполнение полостей,
    разделение соприкасающихся ядер водоразделом. Семена водораздела — h-максимумы карты расстояний:
    ядро делится, только если между частями есть перетяжка глубиной не менее h_um, поэтому
    одиночное овальное ядро не режется пополам. Возвращает метки."""
    ns = ndi.gaussian_filter(norm01(nuc), sigma_um / px)
    m = ns > filters.threshold_otsu(ns)
    r = max(1, int(round(close_um / px)))
    m = ndi.binary_closing(np.pad(m, r), morphology.disk(r))[r:-r, r:-r]
    m = ndi.binary_fill_holes(m)
    m = _rm_small(m, int(20 / px ** 2))
    dist = ndi.gaussian_filter(ndi.distance_transform_edt(m), 1.0 / px)
    markers = measure.label(morphology.h_maxima(dist, h_um / px) & m)
    return segmentation.watershed(-dist, markers, mask=m)


def cell_territories(fg, nuc_lab, px, min_nuc_um2=100):
    """Разделение маски клеток между ядрами: каждый пиксель переднего плана относится к ближайшему ядру
    (водораздел по карте расстояний от ядер, аналог распространения от ядер в CellProfiler).
    Возвращает метки территорий (номер = номер ядра в nuc_lab) и номер центрального ядра."""
    keep = [p.label for p in measure.regionprops(nuc_lab) if p.area * px ** 2 >= min_nuc_um2]
    if not keep:
        return fg.astype(np.int32), 0
    seeds = np.where(np.isin(nuc_lab, keep), nuc_lab, 0)
    terr = segmentation.watershed(ndi.distance_transform_edt(seeds == 0), seeds, mask=fg | (seeds > 0))
    H, W = nuc_lab.shape
    def score(p):   # ближайшее к центру ядро; ядра, касающиеся края кадра, только если других нет
        r0, c0, r1, c1 = p.bbox
        border = r0 == 0 or c0 == 0 or r1 == H or c1 == W
        return (border, np.hypot(p.centroid[0] - H / 2, p.centroid[1] - W / 2))
    central = min((p for p in measure.regionprops(seeds)), key=score).label
    return terr.astype(np.int32), int(central)


def seg_mito_lab(mito, cell=None):
    """Митохондрии по протоколу лаборатории: медианный фильтр, CLAHE, Оцу."""
    m = norm01(ndi.median_filter(mito, 3), mask=cell)
    m = exposure.equalize_adapthist(m, clip_limit=0.02)
    mm = m > filters.threshold_otsu(m[cell] if cell is not None and cell.any() else m)
    if cell is not None:
        mm &= cell
    return _rm_small(mm, 6)


def detect_lysosomes(lyso, cell, nuc_mask, threshold=0.15):
    """Лизосомы: top-hat + лапласиан гауссиана; только внутри клетки и вне ядра. Возвращает координаты (y, x) и радиусы."""
    lt = morphology.white_tophat(ndi.gaussian_filter(lyso, 1), morphology.disk(5))
    blobs = feature.blob_log(norm01(lt, mask=cell), min_sigma=1, max_sigma=3, threshold=threshold)
    keep = [b for b in blobs if cell[int(b[0]), int(b[1])] and not nuc_mask[int(b[0]), int(b[1])]]
    b = np.array(keep).reshape(-1, 3)
    return b[:, :2], b[:, 2] * np.sqrt(2)


NUC_DIAM_UM = 25.0  # типичный диаметр ядра: медианная площадь ядра ~510 мкм² -> эквивалентный диаметр ~25 мкм


def cellpose_masks(arr, model_type='nuclei', gpu=True, diameter_um=None, px=TARGET_PX, model=None):
    """Сегментация Cellpose (версия 3.x). model_type='nuclei' — по каналу ядер, 'cyto3' — клетки (митохондрии + ядра).
    Диаметр задается явно: автоматическая оценка на этих данных принимает митохондрии за клетки,
    а диаметр ядра по умолчанию у Cellpose (~17 пкс) в 8 раз меньше реального при 0,18 мкм/пкс."""
    from cellpose import models
    model = model or models.Cellpose(gpu=gpu, model_type=model_type)
    if model_type == 'nuclei':
        img, channels = norm01(arr[2]), [0, 0]
        diameter_um = diameter_um or NUC_DIAM_UM
    else:
        img, channels = np.stack([norm01(arr[1]), norm01(arr[2])], -1), [1, 2]
    diam = None if diameter_um is None else diameter_um / px
    masks, *_ = model.eval(img, diameter=diam, channels=channels)
    return masks


def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)


def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return (a & b).sum() / max((a | b).sum(), 1)


def object_prf(pred_lab, gt_lab, thr=0.5):
    """Объектные точность, полнота и F1: объект найден, если IoU с эталонным объектом >= thr."""
    P = [r.label for r in measure.regionprops(pred_lab)]
    G = [r.label for r in measure.regionprops(gt_lab)]
    tp = 0
    for g in G:
        gm = gt_lab == g
        cand = np.unique(pred_lab[gm]); cand = cand[cand > 0]
        if any(iou(pred_lab == c, gm) >= thr for c in cand):
            tp += 1
    prec = tp / max(len(P), 1); rec = tp / max(len(G), 1)
    return dict(precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-9))


def central_label(lab):
    """Метка объекта, ближайшего к центру кадра."""
    props = measure.regionprops(lab)
    if not props:
        return 0
    c = np.array(lab.shape) / 2
    return min(props, key=lambda r: np.linalg.norm(np.array(r.centroid) - c)).label


def segment_all(arr, px, nuc_lab=None):
    """Сегментация одного изображения. nuc_lab — готовые метки ядер (например, Cellpose);
    если не заданы, ядра выделяются классическим методом. Анализируется клетка, ядро которой ближе
    всего к центру кадра: ее территория (cell) отделяется от соседних клеток, органеллы ищутся только в ней."""
    fg = seg_cell(arr, px)
    if nuc_lab is None:
        nuc_lab = seg_nuclei_otsu(arr[2], px)
    nuc_lab = nuc_lab.astype(np.int32)
    nuc_mask = nuc_lab > 0
    terr, central = cell_territories(fg, nuc_lab, px)
    cell = (terr == central) if central else fg
    cell = cell | (nuc_lab == central) if central else cell
    mito = seg_mito_lab(arr[1], cell) & ~nuc_mask
    lyso_xy, lyso_r = detect_lysosomes(arr[0], cell, nuc_mask)
    return dict(fg=fg, terr=terr, cell=cell, nuc_lab=nuc_lab, nuc_central=central, mito=mito, lyso_xy=lyso_xy, lyso_r=lyso_r)


# ---------------------------------------------------------------- признаки (без интенсивности)
def _graph_features(skel):
    import networkx as nx
    from skan import Skeleton, summarize
    out = dict(graph_density=np.nan, graph_clustering=np.nan, graph_avg_path=np.nan, n_components=np.nan, cyclomatic=np.nan,
               branch_len_mean=np.nan, n_branches=0)
    if skel.sum() < 3:
        return out
    sk = Skeleton(skel)
    br = summarize(sk, separator='_')
    G = nx.Graph()
    for a, b, l in zip(br['node_id_src'], br['node_id_dst'], br['branch_distance']):
        G.add_edge(int(a), int(b), weight=float(l))
    out['n_branches'] = len(br)
    out['branch_len_mean'] = float(br['branch_distance'].mean())
    out['graph_density'] = nx.density(G)
    out['graph_clustering'] = nx.average_clustering(G)
    comps = list(nx.connected_components(G))
    out['n_components'] = len(comps)
    out['cyclomatic'] = G.number_of_edges() - G.number_of_nodes() + len(comps)
    big = G.subgraph(max(comps, key=len))
    n = big.number_of_nodes()
    if n <= 1:
        out['graph_avg_path'] = 0.0
    elif n <= 300:
        out['graph_avg_path'] = nx.average_shortest_path_length(big)
    else:  # оценка по 200 случайным источникам, иначе слишком долго
        rng = np.random.default_rng(0)
        src = rng.choice(list(big.nodes), 200, replace=False)
        out['graph_avg_path'] = float(np.mean([np.mean(list(nx.single_source_shortest_path_length(big, s).values())[1:]) for s in src]))
    deg = np.array([d for _, d in G.degree()])
    out['frac_endpoints'] = float((deg == 1).mean()); out['frac_junctions'] = float((deg >= 3).mean())
    return out


def _glcm(nuc, mask):
    if mask.sum() < 20:
        return dict(nuc_contrast=np.nan, nuc_homogeneity=np.nan, nuc_entropy=np.nan)
    rr, cc = np.where(mask)
    sub = nuc[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    m = mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    q = (norm01(sub, 1, 99, m) * 31).astype(np.uint8) + 1  # нормировка внутри ядра: текстура, а не яркость
    q[~m] = 0
    g = graycomatrix(q, [1, 2], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=33, symmetric=True)
    g = g[1:, 1:].astype(float)
    g /= np.maximum(g.sum(axis=(0, 1), keepdims=True), 1)
    p = g.mean(axis=(2, 3))
    ent = -np.sum(p[p > 0] * np.log2(p[p > 0]))
    i, j = np.indices(p.shape)
    return dict(nuc_contrast=float((p * (i - j) ** 2).sum()), nuc_homogeneity=float((p / (1 + np.abs(i - j))).sum()), nuc_entropy=float(ent))


def extract_features(arr, px, seg, n_null=500, near_um=0.5, perinuc_um=5.0, rng=0):
    """Традиционные морфологические, топологические и пространственные признаки одного изображения."""
    rng = np.random.default_rng(rng)
    cell, nuc_lab, mito = seg['cell'], seg['nuc_lab'], seg['mito']
    area_cell = max(cell.sum() * px ** 2, 1e-6)
    f = {}
    # митохондрии
    lab = measure.label(mito)
    props = measure.regionprops(lab)
    areas = np.array([p.area for p in props]) * px ** 2
    f['mito_area_frac'] = mito.sum() * px ** 2 / area_cell
    f['mito_n_per_100um2'] = len(props) / area_cell * 100
    f['mito_obj_area_mean'] = float(areas.mean()) if len(areas) else np.nan
    f['mito_obj_area_median'] = float(np.median(areas)) if len(areas) else np.nan
    big = [p for p in props if p.area >= 5]
    f['mito_elongation'] = float(np.mean([p.axis_major_length / max(p.axis_minor_length, 1) for p in big])) if big else np.nan
    f['mito_solidity'] = float(np.mean([p.solidity for p in big])) if big else np.nan
    f['mito_eccentricity'] = float(np.mean([p.eccentricity for p in big])) if big else np.nan
    skel = morphology.skeletonize(mito)
    f['mito_skel_len_per_um2'] = skel.sum() * px / area_cell
    sklab = measure.label(skel, connectivity=2)
    nbr = ndi.convolve(skel.astype(int), np.ones((3, 3), int), mode='constant') - 1
    branched = {l for l in np.unique(sklab[(nbr >= 3) & skel]) if l > 0}
    n_obj = sklab.max()
    f['mito_network_frac'] = len(branched) / max(n_obj, 1)  # доля «сетевых» объектов среди всех
    g = _graph_features(skel)
    g['branch_len_mean'] = g['branch_len_mean'] * px if g['branch_len_mean'] == g['branch_len_mean'] else np.nan
    f.update({'mito_' + k: v for k, v in g.items()})
    # служебные показатели (префикс qc_): в модели не входят, используются для критериев исключения
    def _touch(m): return int(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
    nuc_props = measure.regionprops(nuc_lab)
    f['qc_n_nuclei_field'] = sum(1 for p in nuc_props if p.area * px ** 2 >= 100)  # зависит от размера поля зрения
    f['qc_cell_touches_border'] = _touch(cell)
    f['qc_cell_area_um2'] = float(area_cell)
    # ядро анализируемой (центральной) клетки
    cl = seg.get('nuc_central', None)
    if cl is None:
        cl = central_label(nuc_lab)
    nm = nuc_lab == cl
    f['qc_nuc_found'] = int(bool(cl) and nm.sum() > 0)
    if cl and nm.sum() > 0:
        p = measure.regionprops(nm.astype(int))[0]
        f.update(nuc_area=p.area * px ** 2, nuc_perimeter=p.perimeter * px, nuc_eccentricity=p.eccentricity,
                 nuc_aspect=p.axis_major_length / max(p.axis_minor_length, 1), nuc_solidity=p.solidity,
                 nuc_circularity=4 * np.pi * p.area / max(p.perimeter ** 2, 1), qc_nuc_touches_border=_touch(nm))
        f.update(_glcm(arr[2], nm))
    # лизосомы
    xy = seg['lyso_xy']; r = seg['lyso_r']
    f['lyso_n_per_100um2'] = len(xy) / area_cell * 100
    f['lyso_radius_mean'] = float(r.mean() * px) if len(r) else np.nan
    nuc_mask = nuc_lab > 0
    if len(xy) and nuc_mask.any():
        dnuc = ndi.distance_transform_edt(~nuc_mask) * px
        dn = dnuc[xy[:, 0].astype(int), xy[:, 1].astype(int)]
        f['lyso_dist_nucleus_median'] = float(np.median(dn))
        f['lyso_perinuclear_frac'] = float((dn <= perinuc_um).mean())
    if len(xy) > 1:
        dd, _ = cKDTree(xy).query(xy, k=2)
        f['lyso_nn_dist_median'] = float(np.median(dd[:, 1]) * px)
    # пространственная ассоциация лизосом с митохондриями + нулевая модель
    if len(xy) and mito.any():
        dmito = ndi.distance_transform_edt(~mito) * px
        obs = dmito[xy[:, 0].astype(int), xy[:, 1].astype(int)]
        f['lyso_mito_dist_median'] = float(np.median(obs))
        f['lyso_near_mito_frac'] = float((obs <= near_um).mean())
        allowed = np.argwhere(cell & ~nuc_mask)
        if len(allowed) > len(xy):
            null = []
            for _ in range(n_null):
                pts = allowed[rng.choice(len(allowed), len(xy), replace=False)]
                null.append((dmito[pts[:, 0], pts[:, 1]] <= near_um).mean())
            null = np.array(null)
            f['lyso_near_mito_Z'] = float((f['lyso_near_mito_frac'] - null.mean()) / max(null.std(), 1e-6))
    return f


# ---------------------------------------------------------------- фрагменты для нейросетей
def tiles_for_net(arr, cell, size=224, overlap=0.5, min_frac=0.3, upscale=1, isolate=False, px=TARGET_PX, margin_um=2.0):
    """Фрагменты size x size (каналы нормированы внутри изображения по маске клеток), веса = доля маски во фрагменте.
    isolate=True: сигнал вне анализируемой клетки (с запасом margin_um) обнуляется, чтобы соседние клетки
    не попадали в представление; фон в исходных данных и так обрезан до нуля."""
    a = np.stack([norm01(c, mask=cell) for c in arr])
    if isolate:
        keep = ndi.binary_dilation(cell, iterations=max(1, int(round(margin_um / px))))
        a = a * keep[None]
    m = cell.astype(np.float32)
    if upscale != 1:
        a = np.stack([transform.rescale(c, upscale, order=1) for c in a]); m = transform.rescale(m, upscale, order=0)
    H, W = m.shape
    if H < size or W < size:
        a = np.pad(a, ((0, 0), (0, max(0, size - H)), (0, max(0, size - W)))); m = np.pad(m, ((0, max(0, size - H)), (0, max(0, size - W))))
        H, W = m.shape
    step = int(size * (1 - overlap))
    ys = list(range(0, H - size + 1, step)) or [0]; xs = list(range(0, W - size + 1, step)) or [0]
    tiles, w = [], []
    for y in ys:
        for x in xs:
            frac = m[y:y + size, x:x + size].mean()
            if frac >= min_frac:
                tiles.append(a[:, y:y + size, x:x + size]); w.append(frac)
    if not tiles:  # запасной вариант: центральный фрагмент
        y, x = (H - size) // 2, (W - size) // 2
        tiles.append(a[:, y:y + size, x:x + size]); w.append(1.0)
    return np.stack(tiles).astype(np.float32), np.array(w, np.float32)


# ---------------------------------------------------------------- оценка
def balanced_accuracy(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return np.mean([(yhat[y == c] == c).mean() for c in np.unique(y)])


def hier_bootstrap(meta, B=10000, seed=0):
    """Индексы иерархического бутстрэпа: культуры с возвращением внутри типа (вид × возраст), затем изображения
    с возвращением внутри культуры отдельно для каждой дозы, чтобы сохранить соотношение доз."""
    rng = np.random.default_rng(seed)
    meta = meta.reset_index(drop=True)
    by_type = {t: g.culture.unique() for t, g in meta.groupby('type')}
    idx_by_cult = {c: [g2.index.values for _, g2 in g.groupby('dose')] for c, g in meta.groupby('culture')}
    for _ in range(B):
        idx = []
        for t, cults in by_type.items():
            for c in rng.choice(cults, len(cults), replace=True):
                for ii in idx_by_cult[c]:
                    idx.append(rng.choice(ii, len(ii), replace=True))
        yield np.concatenate(idx)


def linear_cka(X, Y):
    """Центрированное ядерное выравнивание с линейным ядром (Kornblith et al., 2019)."""
    X = X - X.mean(0); Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y, 'fro') ** 2
    return float(hsic / (np.linalg.norm(X.T @ X, 'fro') * np.linalg.norm(Y.T @ Y, 'fro')))


def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i]); adj[i] = min(run, 1)
    return adj


def permute_within(labels, blocks, rng):
    """Перестановка меток внутри блоков (например, пар культура × повтор)."""
    labels = np.asarray(labels).copy(); out = labels.copy()
    for b in np.unique(blocks):
        ii = np.where(blocks == b)[0]
        out[ii] = labels[rng.permutation(ii)]
    return out
