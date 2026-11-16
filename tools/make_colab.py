"""Собирает Colab-ноутбук notebooks/VKR_colab.ipynb. Все ячейки описаны здесь, src/vkr_lib.py встраивается целиком.
Запуск: python tools/make_colab.py           — пересобрать ноутбук
        python tools/make_colab.py --check   — проверить, что закоммиченный ноутбук совпадает с исходниками (для CI)"""
import json, os, sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
LIB = os.path.join(ROOT, 'src', 'vkr_lib.py')
OUT = os.path.join(ROOT, 'notebooks', 'VKR_colab.ipynb')


def cell(kind, src):
    lines = src.strip('\n').split('\n')
    c = {'cell_type': kind, 'metadata': {}, 'source': [l + '\n' for l in lines[:-1]] + [lines[-1]]}
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    return c


def md(s): return cell('markdown', s)
def code(s): return cell('code', s)


C = []

# ============================================================== заголовок, установка, данные
C += [md(r'''
# ВКР: полный конвейер анализа в Google Colab

Один ноутбук на всю работу, потому что отдельные ноутбуки Colab не видят файлы друг друга.

**Перед запуском**
1. Среда выполнения → Сменить среду выполнения → **A100** (или T4). На A100 у Colab около 12 ядер процессора, а модели раздела 6 считаются на процессоре параллельно, поэтому весь ноутбук там проходит примерно за 40–60 минут, на T4 (2 ядра) в несколько раз дольше.
2. Данные (123 TIFF) берутся с Google Диска, из папки исследования: `I — НИР 3/02 — Артефакты исследования/01 Данные/data.zip`. Достаточно одного из вариантов:
   - вставить в ячейку данных ссылку на `data.zip` с доступом «Все, у кого есть ссылка» (`DATA_GDRIVE_URL`), тогда архив скачается без подключения Диска;
   - или подключить свой Google Диск (ячейка сама попросит разрешение): архив или папка с подпапками групп ищутся в папке, название которой содержит `_ВКР_`. Если папку исследования вам прислали ссылкой, сначала добавьте на нее ярлык в «Мой диск»;
   - или перетащить `data.zip` в панель «Файлы» слева.
3. Запустить ячейку установки пакетов: после нее сеанс сам перезапустится (это нормально), дальше запускать все ячейки по порядку (Среда выполнения → Выполнить все ниже). Результаты пишутся в `results/`, последняя ячейка скачивает их архивом.

Ноутбук можно выполнить и вне Colab (например, автоматически): тогда перезапуск сеанса и скачивание архива пропускаются, рабочая папка задается переменной окружения `VKR_WORKDIR`, ссылка на данные — `VKR_DATA_URL`.

| Раздел | Зачем |
|---|---|
| 1. Контроль качества | пригодность и сопоставимость данных, флаги насыщения и расфокусировки |
| 2. Сегментация | ядра Cellpose (основной вариант) и классические, территория центральной клетки, митохондрии, лизосомы, согласие методов; экспорт снимков для ручной разметки |
| 3. Традиционные признаки | базовая модель A в двух вариантах сегментации, автоматические критерии исключения |
| 4. Нейросетевые представления | B — самообученная DINOv2, C — ResNet-50 с учителем (ImageNet) |
| 5. Дообучение ResNet-50 | выигрыш от обучения на собственных данных |
| 6. Сравнение и гипотезы | A, B, C, гибриды, H1–H3, анализы чувствительности, влияние сегментации, батч-эффект, кластеры, кривые обучения |
'''),
      code(r'''
# Установка пакетов. pip может заменить numpy/scipy, поэтому после установки сеанс Colab автоматически перезапускается.
# Увидите сообщение о сбое сеанса: так и задумано, просто запускайте следующую ячейку. Эту ячейку повторно не запускать.
!pip -q install tifffile imagecodecs skan networkx umap-learn gdown "cellpose<4"
!pip check || true
import importlib.util, os
if importlib.util.find_spec('google.colab') is not None and not os.environ.get('VKR_NO_RESTART'):
    os.kill(os.getpid(), 9)
'''),
      code('%%writefile vkr_lib.py\n' + open(LIB, encoding='utf-8').read()),
      code(r'''
# Рабочая папка и данные
import os, sys, glob, zipfile, shutil, json, warnings, importlib.util
warnings.filterwarnings('ignore')
IN_COLAB = importlib.util.find_spec('google.colab') is not None
WORK = '/content' if IN_COLAB else os.path.abspath(os.environ.get('VKR_WORKDIR', os.getcwd()))
os.chdir(WORK); sys.path.insert(0, WORK)
RES = os.path.join(WORK, 'results')
os.makedirs(f'{RES}/figs', exist_ok=True)

# Ссылка на data.zip на Google Диске с доступом «Все, у кого есть ссылка» (вставить после загрузки папки исследования)
DATA_GDRIVE_URL = os.environ.get('VKR_DATA_URL', '')
MOUNT_DRIVE = IN_COLAB          # подключать ли свой Google Диск, если ссылка не задана
DRIVE_PATTERN = '*_ВКР_*'       # так называется папка исследования (Название_ВКР_ФИО_Магистратура)

def find_data(root=WORK):
    # рекурсивно только внутри root/data: в /content/drive смонтирован весь Диск, обходить его целиком долго
    cand = glob.glob(f'{root}/Acom_Old_10Gy') + glob.glob(f'{root}/*/Acom_Old_10Gy') + glob.glob(f'{root}/data/**/Acom_Old_10Gy', recursive=True)
    hits = [h for h in cand if glob.glob(f'{h}/*.tif')]
    return os.path.dirname(sorted(hits)[0]) if hits else None

def unpack(zip_path):
    with zipfile.ZipFile(zip_path) as z: z.extractall(f'{WORK}/data')

DATA = find_data()
if DATA is None and DATA_GDRIVE_URL:                       # 1) скачивание по ссылке
    import gdown
    gdown.download(DATA_GDRIVE_URL, f'{WORK}/data.zip', quiet=False, fuzzy=True)
    unpack(f'{WORK}/data.zip'); DATA = find_data()
if DATA is None and MOUNT_DRIVE:                           # 2) свой Google Диск
    from google.colab import drive
    drive.mount('/content/drive')
    roots = glob.glob(f'/content/drive/MyDrive/{DRIVE_PATTERN}') + glob.glob(f'/content/drive/MyDrive/*/{DRIVE_PATTERN}')
    for r in roots:
        z = sorted(glob.glob(f'{glob.escape(r)}/**/data.zip', recursive=True))
        if z:
            shutil.copy(z[0], f'{WORK}/data.zip'); unpack(f'{WORK}/data.zip'); break
        tifs = [h for h in glob.glob(f'{glob.escape(r)}/**/Acom_Old_10Gy', recursive=True) if glob.glob(f'{h}/*.tif')]
        if tifs:                                           # копия в локальную папку: чтение с Диска медленное
            shutil.copytree(os.path.dirname(tifs[0]), f'{WORK}/data/images', dirs_exist_ok=True); break
    print('папки исследования на Диске:', roots)
    DATA = find_data()
if DATA is None:                                           # 3) архив в рабочей папке или ручная загрузка
    zips = [z for z in glob.glob(f'{WORK}/*.zip') if not z.endswith('results.zip')]
    if not zips and IN_COLAB:
        from google.colab import files
        print('Выберите data.zip с папкой данных')
        zips = [os.path.join(WORK, k) for k in files.upload()]
    if zips: unpack(zips[0])
    DATA = find_data()
assert DATA, 'Не найдена папка с подпапками групп (Acom_Old_10Gy и т. д.)'
print('DATA =', DATA, '| ядер процессора:', os.cpu_count())

import importlib, vkr_lib as V; importlib.reload(V)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from joblib import Parallel, delayed
SEED = 0
np.random.seed(SEED)
META_ALL = V.list_dataset(DATA)          # полный список изображений с путями, не меняется до конца ноутбука
meta = META_ALL.copy()
print(meta.shape, 'культур:', meta.culture.nunique())
assert len(meta) == 123, 'ожидалось 123 изображения'

IMGS = {}   # кэш изображений, приведенных к 0,18 мкм/пкс: file -> arr[3, H, W]
def get_imgs():
    """Читает и приводит к единому масштабу все изображения один раз (параллельно), дальше берет из памяти."""
    if len(IMGS) < len(META_ALL):
        arrs = Parallel(n_jobs=-1)(delayed(lambda p: V.read_image(p)[0])(p) for p in META_ALL.path)
        IMGS.update(dict(zip(META_ALL.file, arrs)))
    return IMGS

SEG_MAIN = 'cellpose'   # основной вариант сегментации ядер; 'classic' — сравнение
def load_mask(file, variant=SEG_MAIN):
    return dict(np.load(f'{RES}/masks/{variant}/{file[:-4]}.npz'))
'''),
      code(r'''
# Если раньше уже скачивали results.zip из этого ноутбука, его можно положить в рабочую папку: результаты восстановятся
if os.path.exists(f'{WORK}/results.zip') and not os.listdir(f'{RES}/figs'):
    with zipfile.ZipFile(f'{WORK}/results.zip') as z: z.extractall(WORK)
    print('результаты восстановлены:', sorted(os.listdir(RES)))
'''),
      code(r'''
# Версии библиотек и оборудование (для раздела «Программная реализация»)
import platform, sklearn, skimage, scipy, torch
try:
    import cellpose; cp_ver = getattr(cellpose, '__version__', 'unknown')
except Exception:
    cp_ver = 'не установлен'
VERS = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
            sklearn=sklearn.__version__, skimage=skimage.__version__, torch=torch.__version__, cellpose=cp_ver,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'нет', cpu_count=os.cpu_count())
json.dump(VERS, open(f'{RES}/versions.json', 'w'), ensure_ascii=False, indent=1); VERS
'''),
      ]

# ============================================================== 1. Контроль качества
C += [md(r'''
# 1. Контроль качества
Размер пикселя, порядок каналов, насыщение и резкость, состав групп, контактные листы для визуального просмотра.
Резкость считается как дисперсия лапласиана **после нормировки** канала по перцентилям, иначе мера зависит от яркости окрашивания и помечает как «расфокусированные» просто тусклые снимки.
Результат: `results/qc_table.csv`. Решение об исключении принимается в разделе 3 по автоматическим критериям (ядро анализируемой клетки целиком в кадре) и вручную через `MANUAL_EXCLUDE`.
'''),
      md('### Состав набора: изображения по группам и повторам'),
      code("pd.crosstab([meta.type, meta.dose], meta.replicate, margins=True)"),
      code(r'''
trip = meta.groupby(['culture', 'replicate']).dose.nunique()
print('полных троек контроль/5/10 Гр в одной культуре и одном повторе:', int((trip == 3).sum()))
'''),
      md('### Показатели качества'),
      code(r'''
qc = pd.DataFrame(Parallel(n_jobs=-1)(delayed(V.qc_metrics)(p) for p in meta.path))
qc = pd.concat([meta, qc], axis=1)
print('файлы с нестандартным порядком каналов:', qc.loc[qc.luts != 'RGB', 'file'].tolist())
qc[['mito_sat_pct', 'lyso_sat_pct', 'nuc_sat_pct', 'mito_focus', 'nuc_focus']].describe().round(2)
'''),
      code(r'''
# Размер пикселя по видам, возрастам и дозам: если он связан с группой, модель может различать группы по масштабу
from scipy import stats
for col in ['species', 'age', 'dose']:
    groups_ = [g.px_um.values for _, g in qc.groupby(col)]
    print(col, qc.groupby(col).px_um.median().round(3).to_dict(), 'Kruskal p =', round(stats.kruskal(*groups_).pvalue, 4))
'''),
      code(r'''
# Флаги для просмотра глазами: насыщение митохондрий > 5% (такие снимки исключаются в анализе чувствительности)
# и наименьшая резкость (5% снимков). Таблица ниже показывает, не связаны ли флаги с видом или дозой.
qc['flag_saturation'] = qc.mito_sat_pct > 5
qc['flag_focus'] = qc.mito_focus < qc.mito_focus.quantile(0.05)
display(qc.groupby(['species', 'irradiated'])[['flag_saturation', 'flag_focus']].sum())
qc.loc[qc.flag_saturation | qc.flag_focus, ['file', 'mito_sat_pct', 'mito_focus', 'flag_saturation', 'flag_focus']]
'''),
      code(r'''
# Ручные решения об исключении после просмотра контактных листов: {'имя файла.tif': 'причина'}
MANUAL_EXCLUDE = {}
'''),
      md('### Контактные листы: строки — доза и культура, столбцы — повтор'),
      code(r'''
from skimage import transform
def thumb(path, size=256):
    arr, px, _ = V.read_image(path, resample=False)
    return transform.resize(V.composite(arr), (size, size), anti_aliasing=True)
THUMBS = dict(zip(meta.file, Parallel(n_jobs=-1)(delayed(thumb)(p) for p in meta.path)))
for t in sorted(meta.type.unique()):
    sub = meta[meta.type == t]
    fig, axes = plt.subplots(9, 4, figsize=(10, 22))
    for ax in axes.ravel(): ax.axis('off')
    for _, r in sub.iterrows():
        i = [0, 5, 10].index(r.dose) * 3 + r.culture_in_type - 1
        axes[i, r.replicate - 1].imshow(THUMBS[r.file]); axes[i, r.replicate - 1].set_title(f'{r.dose} Гр, к{r.culture_in_type}, R{r.replicate}', fontsize=8)
    fig.suptitle(t); plt.tight_layout(); fig.savefig(f'{RES}/figs/sheet_{t}.png', dpi=80); plt.show()
'''),
      code(r'''
qc.drop(columns=['path']).to_csv(f'{RES}/qc_table.csv', index=False)
print('сохранено', f'{RES}/qc_table.csv')
'''),
      ]

# ============================================================== 2. Сегментация
C += [md(r'''
# 2. Сегментация
**Схема.** Ручная разметка запланирована позже. Чтобы получить стартовый результат сейчас, основным методом выделения ядер выбран Cellpose (обобщенная модель `nuclei`, без дообучения), классический метод (Оцу + водораздел) идет как сравнение. Когда появится ручная разметка части снимков, оба метода будут оценены по ней (Dice, IoU, объектные F1), а в разделе 6 будет проверено, меняется ли **результат классификации** при переходе от автоматической сегментации к ручной. Если не меняется, полная ручная разметка не нужна.

1. Все изображения читаются и приводятся к 0,18 мкм/пкс один раз (кэш в памяти).
2. Ядра: Cellpose `nuclei` с явно заданным диаметром 25 мкм (медианная площадь ядра около 510 мкм²) и классический вариант.
3. Территории клеток: передний план (порог треугольника по сумме каналов митохондрий и лизосом) делится между ядрами, анализируется клетка, ядро которой ближе всего к центру кадра. Митохондрии (медиана + CLAHE + Оцу, протокол лаборатории) и лизосомы (top-hat + LoG) ищутся только внутри нее.
4. Сравнение Cellpose и классики для ядер, Cellpose `cyto3` для клеток.
5. Экспорт 18 снимков для ручной разметки и, если разметка уже есть, оценка качества.

Результат: `results/masks/cellpose/*.npz` (основной вариант), `results/masks/classic/*.npz`, `results/seg_agreement.csv`, оверлеи `results/figs/seg_*.png`.
'''),
      code(r'''
# Ядра Cellpose для всех изображений одним пакетом на GPU
from cellpose import models
imgs = get_imgs()
m_nuc = models.Cellpose(gpu=True, model_type='nuclei')
NUC_CP, *_ = m_nuc.eval([V.norm01(imgs[f][2]) for f in META_ALL.file], diameter=V.NUC_DIAM_UM / V.TARGET_PX,
                         channels=[0, 0], batch_size=64)
NUC_CP = dict(zip(META_ALL.file, [m.astype(np.int32) for m in NUC_CP]))
print('изображений без ядер Cellpose:', sum(m.max() == 0 for m in NUC_CP.values()))
'''),
      code(r'''
# Территория центральной клетки, митохондрии и лизосомы в двух вариантах ядер (параллельно на всех ядрах процессора)
for v in ['cellpose', 'classic']: os.makedirs(f'{RES}/masks/{v}', exist_ok=True)
KEYS = ['fg', 'terr', 'cell', 'nuc_lab', 'mito', 'lyso_xy', 'lyso_r']
def seg_one(file, arr, nuc_cp):
    out = []
    for variant, nl in [('cellpose', nuc_cp), ('classic', None)]:
        s = V.segment_all(arr, V.TARGET_PX, nuc_lab=nl)
        np.savez_compressed(f'{RES}/masks/{variant}/{file[:-4]}.npz', nuc_central=s['nuc_central'], **{k: s[k] for k in KEYS})
        out.append(dict(file=file, variant=variant, n_nuclei=int(s['nuc_lab'].max()), central=s['nuc_central'],
                        cell_area_um2=s['cell'].sum() * V.TARGET_PX ** 2, n_lysosomes=len(s['lyso_xy'])))
    return out
rows = Parallel(n_jobs=-1)(delayed(seg_one)(f, imgs[f], NUC_CP[f]) for f in META_ALL.file)
seg_summary = pd.DataFrame([r for rr in rows for r in rr])
seg_summary.groupby('variant')[['n_nuclei', 'cell_area_um2', 'n_lysosomes']].describe().T.round(1)
'''),
      code(r'''
# Согласие Cellpose и классики для ядер: все ядра кадра и ядро анализируемой клетки
rows = []
for f in META_ALL.file:
    a, b = load_mask(f, 'cellpose'), load_mask(f, 'classic')
    ca, cb = a['nuc_lab'] == a['nuc_central'], b['nuc_lab'] == b['nuc_central']
    rows.append(dict(file=f, nuc_dice_all=V.dice(a['nuc_lab'] > 0, b['nuc_lab'] > 0), nuc_iou_all=V.iou(a['nuc_lab'] > 0, b['nuc_lab'] > 0),
                     nuc_dice_central=V.dice(ca, cb), nuc_area_cp=ca.sum() * V.TARGET_PX ** 2, nuc_area_classic=cb.sum() * V.TARGET_PX ** 2,
                     cell_dice_central=V.dice(a['cell'], b['cell'])))
agree = pd.DataFrame(rows); agree.to_csv(f'{RES}/seg_agreement.csv', index=False)
print('снимки с Dice центрального ядра < 0,8:', agree.loc[agree.nuc_dice_central < 0.8, 'file'].tolist())
agree.describe().round(3)
'''),
      code(r'''
# Cellpose cyto3 для клеток (сравнение с территорией центральной клетки). Диаметр задается по площади территории,
# иначе автоматическая оценка принимает митохондрии за клетки. Около 3 минут на A100.
RUN_CP_CYTO = True
if RUN_CP_CYTO:
    m_cyto = models.Cellpose(gpu=True, model_type='cyto3')
    rows = []
    for f in META_ALL.file:
        m = load_mask(f); arr = imgs[f]
        d_px = 2 * np.sqrt(m['cell'].sum() / np.pi)
        cyto, *_ = m_cyto.eval(np.stack([V.norm01(arr[1]), V.norm01(arr[2])], -1), diameter=d_px, channels=[1, 2])
        cn = m['nuc_lab'] == m['nuc_central']
        ids, cnt = np.unique(cyto[cn & (cyto > 0)], return_counts=True)
        cp_cell = cyto == ids[np.argmax(cnt)] if len(ids) else np.zeros_like(cn)
        rows.append(dict(file=f, cell_dice_cyto3=V.dice(m['cell'], cp_cell), n_cells_cyto3=int(cyto.max())))
    cyto_agree = pd.DataFrame(rows); agree = agree.merge(cyto_agree, on='file'); agree.to_csv(f'{RES}/seg_agreement.csv', index=False)
    display(cyto_agree.describe().round(3))
'''),
      code(r'''
# Оверлеи: верхний ряд — Cellpose, нижний — классика. Синий — ядра, белый — ядро анализируемой клетки,
# зеленый — митохондрии, красный — лизосомы, желтый — граница анализируемой клетки
from skimage import segmentation as sg
def overlay(file, variant, ax):
    m = load_mask(file, variant); rgb = V.composite(imgs[file]) * 0.6
    rgb[m['mito']] = [0.1, 0.9, 0.3]
    rgb = sg.mark_boundaries(rgb, m['nuc_lab'], color=(0.3, 0.5, 1))
    rgb = sg.mark_boundaries(rgb, (m['nuc_lab'] == m['nuc_central']).astype(int), color=(1, 1, 1))
    rgb = sg.mark_boundaries(rgb, m['cell'].astype(int), color=(1, 1, 0))
    ax.imshow(rgb); ax.scatter(m['lyso_xy'][:, 1], m['lyso_xy'][:, 0], s=2, c='red'); ax.set_title(f'{file} | {variant}', fontsize=8); ax.axis('off')
worst = agree.nsmallest(2, 'nuc_dice_central').file.tolist()
show_files = meta.groupby('type').file.first().tolist()[:2] + worst
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for j, f in enumerate(show_files):
    overlay(f, 'cellpose', axes[0, j]); overlay(f, 'classic', axes[1, j])
plt.tight_layout(); fig.savefig(f'{RES}/figs/seg_examples.png', dpi=90); plt.show()
'''),
      md(r'''
### Ручная разметка (позже)
Ячейка ниже выгружает 18 снимков (по одному на каждое сочетание типа и дозы и еще 6, стратифицированно по виду и дозе) в масштабе 0,18 мкм/пкс: композит и отдельные каналы Hoechst и MitoTracker. Разметка в Fiji: ядра всех клеток и митохондрии центральной клетки как бинарные маски того же размера, сохранить как `results/manual/<имя файла без .tif>_nuc.png` и `_mito.png`. После этого перезапустить следующую ячейку и раздел 6: качество сегментации и влияние ручной разметки на классификацию посчитаются автоматически.
'''),
      code(r'''
from skimage import io, measure
os.makedirs(f'{RES}/to_annotate', exist_ok=True)
pick = meta.groupby(['type', 'dose']).sample(1, random_state=SEED)
pick = pd.concat([pick, meta.drop(pick.index).groupby(['species', 'dose']).sample(1, random_state=SEED)])
for f in pick.file:
    arr = imgs[f]; stem = f[:-4]
    io.imsave(f'{RES}/to_annotate/{stem}_rgb.png', (V.composite(arr) * 255).astype(np.uint8), check_contrast=False)
    io.imsave(f'{RES}/to_annotate/{stem}_ch_hoechst.png', (V.norm01(arr[2]) * 255).astype(np.uint8), check_contrast=False)
    io.imsave(f'{RES}/to_annotate/{stem}_ch_mito.png', (V.norm01(arr[1]) * 255).astype(np.uint8), check_contrast=False)
print(len(pick), 'изображений в', f'{RES}/to_annotate')
'''),
      code(r'''
# Оценка по ручной разметке и признаки, посчитанные по ручным маскам (если разметка уже есть)
mdir = f'{RES}/manual'; rows, frows = [], []
for fn in (sorted(os.listdir(mdir)) if os.path.isdir(mdir) else []):
    if not fn.endswith('_nuc.png'): continue
    stem = fn[:-8]; file = stem + '.tif'; arr = imgs[file]
    gt_nuc = measure.label(io.imread(f'{mdir}/{fn}') > 0)
    gt_nuc = gt_nuc[:arr.shape[1], :arr.shape[2]]
    for variant in ['cellpose', 'classic']:
        lab = load_mask(file, variant)['nuc_lab']
        r = dict(file=file, object='nucleus', method=variant, dice=V.dice(lab > 0, gt_nuc > 0), iou=V.iou(lab > 0, gt_nuc > 0))
        r.update(V.object_prf(lab, gt_nuc)); rows.append(r)
    s = V.segment_all(arr, V.TARGET_PX, nuc_lab=gt_nuc)            # территория и органеллы от ручных ядер
    if os.path.exists(f'{mdir}/{stem}_mito.png'):
        gm = (io.imread(f'{mdir}/{stem}_mito.png') > 0)[:arr.shape[1], :arr.shape[2]]
        for variant in ['cellpose', 'classic']:
            m = load_mask(file, variant)
            rows.append(dict(file=file, object='mito', method=f'clahe_otsu ({variant} cell)', dice=V.dice(m['mito'], gm & m['cell']), iou=V.iou(m['mito'], gm & m['cell'])))
        s['mito'] = gm & s['cell'] & ~(gt_nuc > 0)
    f_ = V.extract_features(arr, V.TARGET_PX, s); f_['file'] = file; frows.append(f_)
if rows:
    quality = pd.DataFrame(rows); quality.to_csv(f'{RES}/segmentation_quality.csv', index=False)
    pd.DataFrame(frows).to_csv(f'{RES}/features_manual.csv', index=False)
    display(quality.groupby(['object', 'method'])[['dice', 'iou', 'precision', 'recall', 'f1']].mean().round(3))
else:
    print('Ручная разметка пока не найдена в', mdir)
'''),
      ]

# ============================================================== 3. Признаки
C += [md(r'''
# 3. Традиционные признаки
Морфология и граф митохондриальной сети, лизосомы, форма и текстура ядра анализируемой клетки, пространственная ассоциация лизосом с митохондриями (Z по нулевой модели). Все признаки считаются внутри территории центральной клетки.
Абсолютная интенсивность флуоресценции как признак не используется: настройки съемки подбирались под каждый образец. Столбцы с префиксом `qc_` (число ядер в кадре, касание края и т. п.) служебные и в модели не входят.
Признаки считаются для обоих вариантов сегментации: `features.csv` (Cellpose, основной) и `features_classic.csv`.

**Критерии исключения** (автоматически): у анализируемой клетки нет ядра площадью не менее 100 мкм² или ее ядро касается края кадра; плюс ручные решения из `MANUAL_EXCLUDE`. Снимки с насыщением митохондрий > 5% не исключаются, а проверяются в анализе чувствительности.
'''),
      code(r'''
imgs = get_imgs()
def feats(file, arr, variant):
    f = V.extract_features(arr, V.TARGET_PX, load_mask(file, variant), n_null=500)
    f['file'] = file; return f
FEAT = {}
for variant in ['cellpose', 'classic']:
    Fv = pd.DataFrame(Parallel(n_jobs=-1)(delayed(feats)(f, imgs[f], variant) for f in META_ALL.file))
    FEAT[variant] = META_ALL.drop(columns=['path']).merge(Fv, on='file')
    FEAT[variant].to_csv(f'{RES}/features{"" if variant == SEG_MAIN else "_" + variant}.csv', index=False)
F = FEAT[SEG_MAIN]; print(F.shape)
F.describe().T.round(3)
'''),
      code(r'''
# Критерии исключения -> qc_table.csv (столбцы exclude, exclude_reason); раздел 6 их учитывает
qc = pd.read_csv(f'{RES}/qc_table.csv')
Fi = F.set_index('file').loc[qc.file]
reason = np.where(Fi.qc_nuc_found.values == 0, 'нет ядра анализируемой клетки',
         np.where(Fi.qc_nuc_touches_border.fillna(1).values == 1, 'ядро касается края кадра', ''))
reason = [globals().get('MANUAL_EXCLUDE', {}).get(f, r) for f, r in zip(qc.file, reason)]
qc['exclude'] = [bool(r) for r in reason]; qc['exclude_reason'] = reason
qc.to_csv(f'{RES}/qc_table.csv', index=False)
print('исключено:', qc.exclude.sum())
display(qc[qc.exclude][['file', 'exclude_reason']])
qc.groupby(['type', 'dose']).exclude.sum().unstack()
'''),
      code(r'''
# Несколько признаков по группам: вид x доза
show = ['mito_branch_len_mean', 'mito_network_frac', 'mito_graph_clustering', 'nuc_area', 'nuc_circularity', 'nuc_contrast',
        'lyso_n_per_100um2', 'lyso_perinuclear_frac', 'lyso_near_mito_Z']
fig, axes = plt.subplots(3, 3, figsize=(15, 11))
for c, ax in zip(show, axes.ravel()):
    for k, (key, g) in enumerate(F.groupby(['species', 'dose'])):
        ax.boxplot(g[c].dropna(), positions=[k], widths=0.6)
    ax.set_xticks(range(6)); ax.set_xticklabels([f'{s}\n{d} Гр' for s, d in F.groupby(['species', 'dose']).groups], fontsize=8); ax.set_title(c)
plt.tight_layout(); fig.savefig(f'{RES}/figs/features_boxplots.png', dpi=90); plt.show()
'''),
      code(r'''
# Корреляции между признаками и PCA (только для обзора; в моделях прореживание делается внутри обучающей выборки)
ID_COLS = ['dose', 'culture_in_type', 'replicate', 'irradiated']
num = F.select_dtypes('number').drop(columns=ID_COLS + [c for c in F if c.startswith('qc_')])
corr = num.corr().abs()
pairs = [(a, b, round(corr.loc[a, b], 3)) for i, a in enumerate(corr) for b in corr.columns[i + 1:] if corr.loc[a, b] > 0.9]
print('пар с |r| > 0,9:', len(pairs)); print(pairs[:15])
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
Z = PCA(2).fit_transform(StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(num)))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for a, col in zip(ax, ['species', 'dose', 'replicate']):
    for k in sorted(F[col].unique()):
        mk = (F[col] == k).values; a.scatter(Z[mk, 0], Z[mk, 1], s=15, label=str(k))
    a.legend(fontsize=8); a.set_title(f'PCA признаков, цвет: {col}')
plt.show()
'''),
      ]

# ============================================================== 4. Эмбеддинги
C += [md(r'''
# 4. Нейросетевые представления
- DINOv2 ViT-S/14 (основная модель B) и ViT-B/14, самообучение, веса заморожены: каждый канал подается отдельно (дублируется в 3 канала), представление = [CLS, среднее по токенам].
- ResNet-50, обученная с учителем на ImageNet (модель C), веса заморожены: глобальное усреднение последнего блока.
- Фрагменты 224 × 224 пкс при 0,18 мкм/пкс, каналы нормированы по перцентилям внутри клетки, сигнал соседних клеток обнулен; представление изображения = среднее по фрагментам с весами доли клетки во фрагменте.
- Варианты для анализа чувствительности: двукратное увеличение (ViT-S, ×2) и прежняя схема без отделения центральной клетки (ViT-S, все поле).

Основная модель B (ViT-S/14, ×1, центральная клетка) зафиксирована заранее, остальные варианты в проверку гипотез не входят.
Результат: `results/emb_<вариант>.npz` (ключи `files`, `X`).
'''),
      code(r'''
import torch, torchvision
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; print(dev)
torch.manual_seed(SEED)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

def tiles_of(file, upscale=1, isolate=True):
    m = load_mask(file)
    return V.tiles_for_net(imgs[file], m['cell'] if isolate else m['fg'], upscale=upscale, isolate=isolate)

TILES, W = {}, {}   # кэш фрагментов основного варианта (float16), используется и в разделе 5
def build_tiles():
    if not TILES:
        out = Parallel(n_jobs=-1)(delayed(tiles_of)(f) for f in META_ALL.file)
        for f, (t, w) in zip(META_ALL.file, out): TILES[f], W[f] = t.astype(np.float16), w
    return TILES, W
imgs = get_imgs(); build_tiles()
print(sum(len(v) for v in TILES.values()), 'фрагментов')

def load_model(name):
    if name.startswith('dinov2'):
        m = torch.hub.load('facebookresearch/dinov2', name)
    else:
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2); m.fc = torch.nn.Identity()
    return m.eval().to(dev)

@torch.no_grad()
def embed(model, name, tiles, bs=256):
    """tiles: [N, 3 канала, 224, 224]; каждый флуоресцентный канал отдельно -> конкатенация. Смешанная точность на GPU."""
    outs = []
    for ch in range(tiles.shape[1]):
        parts = []
        for i in range(0, len(tiles), bs):
            x = torch.from_numpy(np.ascontiguousarray(tiles[i:i + bs, ch:ch + 1])).to(dev).float().repeat(1, 3, 1, 1)
            x = (x - MEAN) / STD
            with torch.autocast('cuda', dtype=torch.float16, enabled=dev == 'cuda'):
                if name.startswith('dinov2'):
                    o = model.forward_features(x); z = torch.cat([o['x_norm_clstoken'], o['x_norm_patchtokens'].mean(1)], 1)
                else:
                    z = model(x)
            parts.append(z.float().cpu().numpy())
        outs.append(np.concatenate(parts))
    return np.concatenate(outs, 1)

def run(tag, name, upscale=1, isolate=True, chunk=24):
    """Эмбеддинги всех изображений: фрагменты всех снимков идут на GPU большими пакетами."""
    model = load_model(name); X = []
    files = META_ALL.file.tolist()
    for s in range(0, len(files), chunk):
        fs = files[s:s + chunk]
        if upscale == 1 and isolate:
            tw = [(TILES[f], W[f]) for f in fs]
        else:
            tw = Parallel(n_jobs=-1)(delayed(tiles_of)(f, upscale, isolate) for f in fs)
        allt = np.concatenate([t for t, _ in tw]); E = embed(model, name, allt); k = 0
        for t, w in tw:
            e = E[k:k + len(t)]; k += len(t); X.append((e * w[:, None]).sum(0) / w.sum())
    X = np.stack(X).astype(np.float32)
    np.savez_compressed(f'{RES}/emb_{tag}.npz', files=np.array(files), X=X)
    print('saved', tag, X.shape); del model; torch.cuda.empty_cache()
'''),
      code(r'''
run('dinov2_vits14_x1', 'dinov2_vits14')          # B, основная
run('resnet50_x1', 'resnet50')                    # C
'''),
      code(r'''
# Варианты для анализа чувствительности
run('dinov2_vitb14_x1', 'dinov2_vitb14')
run('dinov2_vits14_x2', 'dinov2_vits14', upscale=2)
run('dinov2_vits14_field', 'dinov2_vits14', isolate=False)
'''),
      ]

# ============================================================== 5. Дообучение
C += [md(r'''
# 5. Дообучение ResNet-50
Для каждой из 12 культур: обучение на фрагментах остальных культур (обновляются последний сверточный блок и классификатор), аугментации — повороты на 90°, отражения, изменение яркости каналов и гауссов шум. Число эпох фиксировано (8), ранней остановки нет: при 11 обучающих культурах отдельная внутренняя выборка для остановки оставила бы слишком мало данных. Предсказание для изображения = среднее вероятностей фрагментов с весами доли клетки. Зерна генераторов фиксированы для каждого блока.
Все фрагменты хранятся на GPU, обучение в смешанной точности: около 10–15 минут на A100, 40–60 минут на T4.
Результат `results/oof_resnet_ft.csv` подхватывается разделом 6.
'''),
      code(r'''
RUN_FINETUNE = True
if RUN_FINETUNE:
    import torch.nn as nn
    build_tiles()
    excl = set(pd.read_csv(f'{RES}/qc_table.csv').query('exclude').file)
    ft_files = [f for f in META_ALL.file if f not in excl]
    mi = META_ALL.set_index('file').loc[ft_files]
    XT = torch.from_numpy(np.concatenate([TILES[f] for f in ft_files])).to(dev)          # float16 на GPU
    owner = np.concatenate([[i] * len(TILES[f]) for i, f in enumerate(ft_files)])
    lab_img = mi.irradiated.values; cult_img = mi.culture.values

    def augment(x):
        if torch.rand(1) < .5: x = torch.flip(x, [3])
        if torch.rand(1) < .5: x = torch.flip(x, [2])
        x = torch.rot90(x, int(torch.randint(4, (1,))), [2, 3])
        return x * (0.9 + 0.2 * torch.rand(len(x), 3, 1, 1, device=dev)) + 0.02 * torch.randn_like(x)

    def train_fold(tr_img, seed, epochs=8, bs=64):
        torch.manual_seed(seed); np.random.seed(seed)
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2); m.fc = nn.Linear(2048, 2)
        for name, p in m.named_parameters(): p.requires_grad = name.startswith('layer4') or name.startswith('fc')
        m = m.to(dev).to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-2)
        scaler = torch.amp.GradScaler('cuda')
        idx = np.where(np.isin(owner, tr_img))[0]; idx_t = torch.from_numpy(idx).to(dev)
        Y = torch.from_numpy(lab_img[owner[idx]]).to(dev)
        cw = torch.tensor(len(Y) / (2 * np.bincount(lab_img[owner[idx]])), dtype=torch.float32, device=dev)
        for ep in range(epochs):
            m.train(); perm = torch.randperm(len(idx), device=dev)
            for i in range(0, len(idx), bs):
                b = perm[i:i + bs]; x = augment(XT[idx_t[b]].float()); x = ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)
                with torch.autocast('cuda', dtype=torch.float16):
                    loss = nn.functional.cross_entropy(m(x), Y[b], weight=cw)
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        return m.eval()

    @torch.no_grad()
    def predict(m, i):
        x = ((XT[torch.from_numpy(np.where(owner == i)[0]).to(dev)].float() - MEAN) / STD).contiguous(memory_format=torch.channels_last)
        with torch.autocast('cuda', dtype=torch.float16):
            pr = torch.softmax(m(x).float(), 1)[:, 1].cpu().numpy()
        w = W[ft_files[i]]; return float((pr * w).sum() / w.sum())

    rows = []
    for k, c in enumerate(sorted(set(cult_img))):
        te = np.where(cult_img == c)[0]; tr = np.where(cult_img != c)[0]
        m = train_fold(tr, seed=SEED + k)
        rows += [dict(file=ft_files[i], p=predict(m, i)) for i in te]; print(c, 'готово')
        del m; torch.cuda.empty_cache()
    oof = pd.DataFrame(rows); oof.to_csv(f'{RES}/oof_resnet_ft.csv', index=False)
    yy = META_ALL.set_index('file').loc[oof.file].irradiated.values
    print('BA дообученной ResNet-50 =', round(V.balanced_accuracy(yy, (oof.p.values >= .5).astype(int)), 3))
    del XT; torch.cuda.empty_cache()
'''),
      ]

# ============================================================== 6. Оценка
C += [md(r'''
# 6. Сравнение представлений и проверка гипотез
Задача: облученные (5 и 10 Гр) против контрольных. Перекрестная проверка с исключением одной культуры (LOCO, 12 блоков), подбор C по внутренней проверке, сгруппированной по культурам. Блоки считаются параллельно на всех ядрах процессора.
- A — традиционные признаки (сегментация Cellpose), B — DINOv2 (самообучение), C — ResNet-50 ImageNet (с учителем), D — гибриды: D1 (A + главные компоненты B), D2 (стекинг A, B, C), D_avg (среднее вероятностей A, B, C без обучения весов).
- H1: BA(B) − BA(A) ≥ 10 п. п., нижняя граница 95% ДИ > 0.
- H2: остаток B после удаления части, предсказуемой по A: BA − 50% ≥ 10 п. п., перестановочный тест (1000 перестановок).
- H3: сдвиг относительно контроля той же культуры и повтора, R = Δ(Acomys) / Δ(Mus) ≤ 0,5, перестановка видов между культурами.
- Анализы чувствительности: схемы валидации, исключение насыщенных снимков, варианты H3, влияние сегментации (классика, позже ручная разметка), представление всего поля.
- Дополнительно: kNN, положительный контроль (вид, возраст), батч-эффект, кластеризация, UMAP, кривые обучения линейного классификатора.
'''),
      code(r'''
import itertools
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

ID_COLS = ['dose', 'culture_in_type', 'replicate', 'irradiated']
qc = pd.read_csv(f'{RES}/qc_table.csv')
excl = set(qc.loc[qc.exclude.astype(bool), 'file']) if 'exclude' in qc else set()
F = pd.read_csv(f'{RES}/features.csv'); F = F[~F.file.isin(excl)].reset_index(drop=True)
meta = F[['file', 'species', 'age', 'type', 'dose', 'culture', 'culture_in_type', 'replicate', 'irradiated']].merge(
    META_ALL[['file', 'path']], on='file').reset_index(drop=True)
feat_cols = [c for c in F.select_dtypes('number').columns if c not in ID_COLS and not c.startswith('qc_')]
A = F[feat_cols].values
A_cl = pd.read_csv(f'{RES}/features_classic.csv').set_index('file').loc[meta.file, feat_cols].values
SAT = qc.set_index('file').loc[meta.file, 'mito_sat_pct'].values

def load_emb(name):
    z = np.load(f'{RES}/emb_{name}.npz', allow_pickle=True)
    d = dict(zip(z['files'], z['X'])); return np.stack([d[f] for f in meta.file])
EMB = {n: load_emb(n) for n in ['dinov2_vits14_x1', 'dinov2_vitb14_x1', 'dinov2_vits14_x2', 'dinov2_vits14_field', 'resnet50_x1']
       if os.path.exists(f'{RES}/emb_{n}.npz')}
y = meta.irradiated.values; groups = meta.culture.values
B_NAME = 'dinov2_vits14_x1'; C_NAME = 'resnet50_x1'
print(len(meta), 'изображений (исключено', len(excl), ');', len(feat_cols), 'признаков;', {k: v.shape for k, v in EMB.items()})
'''),
      md('### Протокол: обучение в блоках, подбор C по внутренней проверке, сгруппированной по культурам'),
      code(r'''
CS = (0.001, 0.01, 0.1, 1, 10)

class CorrPrune(BaseEstimator, TransformerMixin):
    """Удаляет признаки с |r| > thr, параметры оцениваются только на обучающей части."""
    def __init__(self, thr=0.9): self.thr = thr
    def fit(self, X, y=None):
        with np.errstate(invalid='ignore', divide='ignore'):
            c = np.abs(np.corrcoef(X, rowvar=False))
        keep = []
        for j in range(X.shape[1]):
            if all(c[j, k] <= self.thr for k in keep if np.isfinite(c[j, k])): keep.append(j)
        self.keep_ = keep; return self
    def transform(self, X): return X[:, self.keep_]

class Concat(BaseEstimator, TransformerMixin):
    """Гибрид D1: признаки (импутация, масштаб, прореживание) + главные компоненты эмбеддинга, все оценивается на обучающей части."""
    def __init__(self, n_a=0): self.n_a = n_a
    def fit(self, X, y=None):
        a, b = X[:, :self.n_a], X[:, self.n_a:]
        self.pa = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), CorrPrune(0.9)).fit(a)
        self.pb = make_pipeline(Normalizer(), StandardScaler(), PCA(0.95, svd_solver='full')).fit(b); return self
    def transform(self, X):
        return np.hstack([self.pa.transform(X[:, :self.n_a]), self.pb.transform(X[:, self.n_a:])])

def make_model(kind, n_a=None):
    lr = LogisticRegression(class_weight='balanced', max_iter=5000)
    if kind == 'features':
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), CorrPrune(0.9), lr)
    if kind == 'concat':
        return make_pipeline(Concat(n_a), StandardScaler(), lr)
    return make_pipeline(Normalizer(), StandardScaler(), PCA(0.95, svd_solver='full'), lr)

def fit_fold(X, yv, tr, te, kind, Cs, w, g_inner, n_a):
    n_inner = min(4, len(np.unique(g_inner[tr])))
    gs = GridSearchCV(make_model(kind, n_a), {'logisticregression__C': list(Cs)},
                      cv=list(GroupKFold(n_inner).split(X[tr], yv[tr], g_inner[tr])), scoring='balanced_accuracy', n_jobs=1)
    gs.fit(X[tr], yv[tr], **({'logisticregression__sample_weight': w[tr]} if w is not None else {}))
    return te, gs.predict_proba(X[te])[:, 1]

def cv_proba(X, yv, splits, kind, Cs=CS, w=None, g_inner=None, n_a=None, n_jobs=-1):
    """Вероятности вне выборки для произвольного набора разбиений (tr, te); подбор C — внутри обучающей части по культурам."""
    g_inner = groups if g_inner is None else g_inner
    out = Parallel(n_jobs=n_jobs)(delayed(fit_fold)(X, yv, tr, te, kind, Cs, w, g_inner, n_a) for tr, te in splits)
    p = np.full(len(yv), np.nan)
    for te, pp in out: p[te] = pp
    return p

def loco_splits(g): return [(g != c, g == c) for c in np.unique(g)]
LOCO = loco_splits(groups)
def loco_proba(X, yv=None, kind='emb', **kw): return cv_proba(X, y if yv is None else yv, LOCO, kind, **kw)

OOF = {'A_features': loco_proba(A, kind='features'), 'A_features_classic': loco_proba(A_cl, kind='features')}
for n, X in EMB.items():
    OOF[n] = loco_proba(X)
if os.path.exists(f'{RES}/oof_resnet_ft.csv'):
    ft = pd.read_csv(f'{RES}/oof_resnet_ft.csv').set_index('file')
    if set(meta.file) <= set(ft.index): OOF['resnet50_finetuned'] = ft.loc[meta.file].p.values
print('готово:', list(OOF))
'''),
      code(r'''
# Гибриды. D1: склейка признаков и главных компонент DINOv2.
OOF['D1_concat_A_B'] = loco_proba(np.hstack([A, EMB[B_NAME]]), kind='concat', Cs=(0.001, 0.01, 0.1, 1), n_a=A.shape[1])

# D2: стекинг. Метамодель учится на вероятностях базовых моделей, полученных по внутренней LOCO внутри обучающей части
# (на снимках, которых базовая модель не видела). Внутренние вероятности считаются один раз на пару (база, блок)
# и переиспользуются в вариантах абляции.
BASES = {'A': (A, 'features'), 'B': (EMB[B_NAME], 'emb'), 'C': (EMB[C_NAME], 'emb')}
OUTER = {'A': OOF['A_features'], 'B': OOF[B_NAME], 'C': OOF[C_NAME]}
def inner_oof(X, kind, tr):
    sub = np.where(tr)[0]; g = groups[sub]
    return cv_proba(X[sub], y[sub], loco_splits(g), kind, g_inner=g, n_jobs=1)
jobs = [(b, k) for b in BASES for k in range(len(LOCO))]
res = Parallel(n_jobs=-1)(delayed(inner_oof)(BASES[b][0], BASES[b][1], LOCO[k][0]) for b, k in jobs)
ZTR = dict(zip(jobs, res))
def stack(names):
    p = np.zeros(len(y))
    for k, (tr, te) in enumerate(LOCO):
        Ztr = np.column_stack([ZTR[(b, k)] for b in names]); Zte = np.column_stack([OUTER[b][te] for b in names])
        p[te] = LogisticRegression(class_weight='balanced').fit(Ztr, y[tr]).predict_proba(Zte)[:, 1]
    return p
OOF['D2_stack_ABC'] = stack(['A', 'B', 'C'])
for drop in BASES:   # абляция: убираем по одной базовой модели
    OOF[f'D2_stack_without_{drop}'] = stack([b for b in BASES if b != drop])
# D_avg: простое среднее вероятностей без обучения весов (базовая линия для стекинга)
OOF['D_avg_ABC'] = np.mean([OUTER[b] for b in 'ABC'], 0)
'''),
      code(r'''
BOOT = list(V.hier_bootstrap(meta, B=10000, seed=SEED))
def ba(p, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    return V.balanced_accuracy(y[idx], (p[idx] >= 0.5).astype(int))
def boot_ci(stat):
    s = np.array([stat(i) for i in BOOT]); return np.percentile(s, 2.5), np.percentile(s, 97.5), s
rows = []
for n, p in OOF.items():
    lo, hi, _ = boot_ci(lambda i: ba(p, i))
    rows.append(dict(model=n, BA=ba(p), CI_low=lo, CI_high=hi, AUC=roc_auc_score(y, p)))
results = pd.DataFrame(rows).round(3); results.to_csv(f'{RES}/results_models.csv', index=False); results
'''),
      code(r'''
# BA по видам и возрастам и по дозам (для основной и базовой моделей)
for n in ['A_features', B_NAME]:
    pred = (OOF[n] >= 0.5).astype(int)
    print(n, {t: round(V.balanced_accuracy(y[meta.type == t], pred[meta.type == t]), 3) for t in sorted(meta.type.unique())})
    print('   доля верно распознанных по дозам:', {d: round((pred[meta.dose == d] == y[meta.dose == d]).mean(), 3) for d in [0, 5, 10]})
'''),
      md('### Гипотеза 1'),
      code(r'''
pA, pB = OOF['A_features'], OOF[B_NAME]
d_obs = ba(pB) - ba(pA)
lo1, hi1, d_bs = boot_ci(lambda i: ba(pB, i) - ba(pA, i))
p1 = (np.sum(d_bs <= 0) + 1) / (len(d_bs) + 1)
print(f'Δ1 = {d_obs*100:.1f} п. п., 95% ДИ [{lo1*100:.1f}; {hi1*100:.1f}], односторонний p = {p1:.4f}')
print('доля несовпадающих предсказаний:', round(((pA >= .5) != (pB >= .5)).mean(), 3))
'''),
      md('### Гипотеза 2'),
      code(r'''
def prep(kind):
    """Предобработка блока: признаки — импутация и масштаб; эмбеддинги — L2, масштаб и PCA(95%)."""
    if kind == 'features': return make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
    return make_pipeline(Normalizer(), StandardScaler(), PCA(0.95, svd_solver='full'))

def residual_ba(yv, Xa, Xb, n_inner=4, kind_a='features', kind_b='emb'):
    """LOCO: предобработка и ĥ: Xa -> Xb учатся на обучающих культурах; остатки для обучения получаются вне выборки по внутренним блокам."""
    p = np.zeros(len(yv))
    for tr, te in LOCO:
        pa = prep(kind_a).fit(Xa[tr]); pb = prep(kind_b).fit(Xb[tr])
        a_tr, a_te, b_tr, b_te = pa.transform(Xa[tr]), pa.transform(Xa[te]), pb.transform(Xb[tr]), pb.transform(Xb[te])
        r_tr = np.zeros_like(b_tr)
        for itr, ite in GroupKFold(n_inner).split(a_tr, groups=groups[tr]):
            h = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(a_tr[itr], b_tr[itr]); r_tr[ite] = b_tr[ite] - h.predict(a_tr[ite])
        h = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(a_tr, b_tr); r_te = b_te - h.predict(a_te)
        clf = LogisticRegression(class_weight='balanced', C=0.1, max_iter=5000).fit(r_tr, yv[tr]); p[te] = clf.predict_proba(r_te)[:, 1]
    return V.balanced_accuracy(yv, (p >= 0.5).astype(int)), p

ba_res, p_res = residual_ba(y, A, EMB[B_NAME])
_a = prep('features').fit_transform(A); _b = prep('emb').fit_transform(EMB[B_NAME])
print('R² (A -> PCA(B)) на всех данных, для справки =', round(RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(_a, _b).score(_a, _b), 3))
print('линейная CKA (A, PCA(B)) =', round(V.linear_cka(_a, _b), 3))
ba_rev, _ = residual_ba(y, EMB[B_NAME], A, kind_a='emb', kind_b='features')
print(f'обратный анализ: остаток традиционных признаков после удаления части, предсказуемой по B: BA = {ba_rev:.3f}')
N_PERM = 1000
blocks = (meta.culture + '_R' + meta.replicate.astype(str)).values
perm_seeds = np.random.SeedSequence(SEED).spawn(N_PERM)
def h2_chunk(seeds, Xa, Xb, yv, blk):
    return [residual_ba(V.permute_within(yv, blk, np.random.default_rng(s)), Xa, Xb)[0] for s in seeds]
chunks = np.array_split(np.arange(N_PERM), 4 * (os.cpu_count() or 1))
perm = np.concatenate(Parallel(n_jobs=-1)(delayed(h2_chunk)([perm_seeds[i] for i in ch], A, EMB[B_NAME], y, blocks) for ch in chunks if len(ch)))
p2 = (np.sum(perm >= ba_res) + 1) / (N_PERM + 1)
print(f'BA_res = {ba_res:.3f} (превышение {100*(ba_res-0.5):.1f} п. п.), перестановочный p = {p2:.4f}')
'''),
      md('### Гипотеза 3'),
      code(r'''
# Показатель облучения t: логит вероятности модели на DINOv2, обученной с одинаковым вкладом видов
w_sp = np.where(meta.species == 'Acom', 0.5 / (meta.species == 'Acom').mean(), 0.5 / (meta.species == 'Mus').mean())
p_t = loco_proba(EMB[B_NAME], w=w_sp)
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / np.clip(1 - p, 1e-4, 1))
t = logit(p_t)

BLK = pd.factorize(blocks)[0]
CULT_OF = pd.Series(meta.culture.values).groupby(BLK).first().values
CINFO = meta.drop_duplicates('culture').set_index('culture')[['species', 'age']]
def shifts(tv, yv, sel=None):
    """Δ по культурам: среднее по повторам от (среднее t облученных − t контроля) в паре культура × повтор."""
    sel = np.ones(len(tv), bool) if sel is None else sel; out = {}
    for b in range(BLK.max() + 1):
        ii = (BLK == b) & sel; irr, ctl = ii & (yv == 1), ii & (yv == 0)
        if irr.any() and ctl.any(): out.setdefault(CULT_OF[b], []).append(tv[irr].mean() - tv[ctl].mean())
    S = CINFO.loc[list(out)].assign(delta=[np.mean(v) for v in out.values()])
    return S

def h3_test(S, agg=np.mean):
    """R и односторонний точный перестановочный тест: видовая принадлежность культур переставляется внутри возрастных групп."""
    dA, dM = agg(S.delta[S.species == 'Acom']), agg(S.delta[S.species == 'Mus'])
    obs = dM - dA; strata = [S.index[S.age == a] for a in sorted(S.age.unique())]
    combos = [list(itertools.combinations(s, (S.loc[s].species == 'Mus').sum())) for s in strata]
    null = []
    for pick in itertools.product(*combos):
        mus = set().union(*map(set, pick)); lab = np.array(['Mus' if c in mus else 'Acom' for c in S.index])
        null.append(agg(S.delta[lab == 'Mus']) - agg(S.delta[lab == 'Acom']))
    return dict(delta_Acomys=dA, delta_Mus=dM, R=dA / dM if dM > 0 else np.nan, p=np.mean(np.array(null) >= obs - 1e-12), n_perm=len(null))

S = shifts(t, y); h3 = h3_test(S); R, p3 = h3['R'], h3['p']
print(S.round(3)); print({k: round(v, 4) for k, v in h3.items()})
# Значим ли сам сдвиг у Mus: метки доз переставляются внутри пары культура × повтор
rng = np.random.default_rng(SEED + 1); dM = h3['delta_Mus']
null_m = [shifts(t, V.permute_within(y, blocks, rng)).query("species == 'Mus'").delta.mean() for _ in range(2000)]
p_mus = (np.sum(np.array(null_m) >= dM) + 1) / 2001
print(f'p (Δ Mus > Δ Acomys) = {p3:.4f}; p (Δ Mus > 0) = {p_mus:.4f}')
'''),
      code(r'''
# Итог по гипотезам с поправкой Холма
adj = V.holm([p1, p2, p3])
summary = pd.DataFrame([
    dict(hypothesis='H1', effect=f'{d_obs*100:.1f} п. п.', threshold='≥ 10 п. п.', p=p1, p_holm=adj[0], CI_low=lo1 * 100,
         confirmed=bool(d_obs >= 0.10 and lo1 > 0 and adj[0] < 0.05)),
    dict(hypothesis='H2', effect=f'{(ba_res-0.5)*100:.1f} п. п.', threshold='≥ 10 п. п.', p=p2, p_holm=adj[1], CI_low=np.nan,
         confirmed=bool(ba_res - 0.5 >= 0.10 and adj[1] < 0.05)),
    dict(hypothesis='H3', effect=f'R = {R:.2f}', threshold='R ≤ 0,5', p=p3, p_holm=adj[2], CI_low=np.nan,
         confirmed=bool(R == R and R <= 0.5 and adj[2] < 0.05 and p_mus < 0.05)),
])
summary.to_csv(f'{RES}/hypotheses.csv', index=False); summary
'''),
      md(r'''
### Анализ чувствительности для H3
Основной анализ (логит, среднее по культурам) зафиксирован заранее. Логит не ограничен, поэтому одна культура с очень уверенными предсказаниями может определять R почти целиком; варианты ниже показывают, насколько вывод устойчив: шкала вероятностей, медиана по культурам, отдельно дозы и возрасты, показатель t по традиционным признакам, исключение каждой культуры по очереди.
'''),
      code(r'''
rows = [dict(variant='основной: логит, среднее', **h3)]
rows.append(dict(variant='шкала вероятностей', **h3_test(shifts(p_t, y))))
rows.append(dict(variant='медиана по культурам', **h3_test(S, agg=np.median)))
for d in [5, 10]:
    rows.append(dict(variant=f'только {d} Гр', **h3_test(shifts(t, y, sel=meta.dose.isin([0, d]).values))))
for a in ['Young', 'Old']:
    rows.append(dict(variant=f'только {a}', **h3_test(S[S.age == a])))
p_tA = loco_proba(A, kind='features', w=w_sp)
rows.append(dict(variant='t по традиционным признакам', **h3_test(shifts(logit(p_tA), y))))
for c in S.index:
    rows.append(dict(variant=f'без {c}', **h3_test(S.drop(c))))
h3_sens = pd.DataFrame(rows).round(3); h3_sens.to_csv(f'{RES}/h3_sensitivity.csv', index=False); h3_sens
'''),
      md(r'''
### Анализ чувствительности: схемы валидации и исключение насыщенных снимков
Для A и B: три блока (в каждом по одной культуре каждого сочетания вида и возраста), исключение повтора целиком, двойное исключение (обучение без культуры c и без повтора r, проверка на паре c × r), а также LOCO без снимков с насыщением митохондрий > 5%.
'''),
      code(r'''
rep = meta.replicate.values; cit = meta.culture_in_type.values
SCHEMES = {'LOCO (основная)': LOCO,
           'три блока': loco_splits(cit),
           'исключение повтора': loco_splits(rep),
           'двойное исключение': [((groups != c) & (rep != r), (groups == c) & (rep == r))
                                  for c, r in meta[['culture', 'replicate']].drop_duplicates().itertuples(index=False)]}
rows = []
for name, splits in SCHEMES.items():
    pa_ = OOF['A_features'] if name.startswith('LOCO') else cv_proba(A, y, splits, 'features')
    pb_ = OOF[B_NAME] if name.startswith('LOCO') else cv_proba(EMB[B_NAME], y, splits, 'emb')
    rows.append(dict(scheme=name, n_splits=len(splits), BA_A=ba(pa_), BA_B=ba(pb_), delta_pp=100 * (ba(pb_) - ba(pa_))))
keep = SAT <= 5
sub = np.where(keep)[0]; g_sub = groups[sub]
pa_ = cv_proba(A[sub], y[sub], loco_splits(g_sub), 'features', g_inner=g_sub)
pb_ = cv_proba(EMB[B_NAME][sub], y[sub], loco_splits(g_sub), 'emb', g_inner=g_sub)
bal = lambda p: V.balanced_accuracy(y[sub], (p >= .5).astype(int))
rows.append(dict(scheme=f'LOCO без насыщенных (n = {keep.sum()})', n_splits=len(np.unique(g_sub)), BA_A=bal(pa_), BA_B=bal(pb_), delta_pp=100 * (bal(pb_) - bal(pa_))))
sens = pd.DataFrame(rows).round(3); sens.to_csv(f'{RES}/validation_sensitivity.csv', index=False); sens
'''),
      md(r'''
### Влияние сегментации на классификацию
Нужна ли ручная разметка, решается по тому, меняет ли она результат. Сейчас сравниваются признаки по ядрам Cellpose (основной вариант) и по классическим ядрам; когда появится `features_manual.csv` (раздел 2), признаки размеченных снимков заменяются на посчитанные по ручным маскам и модель A переобучается.
Правило решения задано заранее: если различие BA модели A между вариантами сегментации меньше 5 п. п. (половина порога H1) и 95% ДИ различия включает 0, сегментация считается не влияющей на вывод, и полная ручная разметка не требуется; качество масок (Dice, IoU) по размеченной подвыборке все равно приводится.
'''),
      code(r'''
from scipy.stats import spearmanr
def seg_compare(p_other, label):
    d = ba(OOF['A_features']) - ba(p_other); lo, hi, _ = boot_ci(lambda i: ba(OOF['A_features'], i) - ba(p_other, i))
    return dict(comparison=label, BA_main=ba(OOF['A_features']), BA_other=ba(p_other), delta_pp=100 * d, CI_low_pp=100 * lo, CI_high_pp=100 * hi,
                pred_agreement=((OOF['A_features'] >= .5) == (p_other >= .5)).mean(), negligible=bool(abs(d) < 0.05 and lo <= 0 <= hi))
rows = [seg_compare(OOF['A_features_classic'], 'Cellpose vs классика')]
rho = {c: spearmanr(A[:, j], A_cl[:, j], nan_policy='omit')[0] for j, c in enumerate(feat_cols)}
if os.path.exists(f'{RES}/features_manual.csv'):
    Fm = pd.read_csv(f'{RES}/features_manual.csv').set_index('file')
    Fm = Fm.loc[[f for f in Fm.index if f in set(meta.file)]]
    A_man = A.copy(); pos = meta.reset_index().set_index('file').loc[Fm.index, 'index'].values
    A_man[pos] = Fm.reindex(columns=feat_cols).values
    OOF['A_features_manual_subset'] = loco_proba(A_man, kind='features')
    rows.append(seg_compare(OOF['A_features_manual_subset'], f'Cellpose vs ручная разметка ({len(Fm)} снимков)'))
    rho_man = {c: spearmanr(A[pos, j], A_man[pos, j], nan_policy='omit')[0] for j, c in enumerate(feat_cols)}
    display(pd.DataFrame({'rho_cellpose_vs_classic': rho, 'rho_cellpose_vs_manual': rho_man}).round(3))
seg_impact = pd.DataFrame(rows).round(3); seg_impact.to_csv(f'{RES}/segmentation_impact.csv', index=False)
print('признаки с наименьшим согласием Cellpose и классики:', sorted(rho.items(), key=lambda kv: kv[1])[:6])
seg_impact
'''),
      md('### Дополнительный анализ: kNN, положительный контроль (вид, возраст)'),
      code(r'''
def loco_knn(X, yv, k=5):
    def one(tr, te):
        m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Normalizer(), KNeighborsClassifier(k)).fit(X[tr], yv[tr]); return te, m.predict(X[te])
    pred = np.zeros(len(yv), int)
    for te, pp in Parallel(n_jobs=-1)(delayed(one)(tr, te) for tr, te in LOCO): pred[te] = pp
    return V.balanced_accuracy(yv, pred)
reps = {'A_features': A} | EMB
ys = (meta.species == 'Mus').astype(int).values; ya = (meta.age == 'Old').astype(int).values
extra = []
for n, X in reps.items():
    kind = 'features' if n == 'A_features' else 'emb'
    extra.append(dict(rep=n, knn_irradiated=loco_knn(X, y), species_BA=V.balanced_accuracy(ys, (loco_proba(X, ys, kind) >= .5).astype(int)),
                      age_BA=V.balanced_accuracy(ya, (loco_proba(X, ya, kind) >= .5).astype(int))))
pd.DataFrame(extra).round(3)
'''),
      md('### Батч-эффект: предсказывается ли номер повтора внутри одной группы (вид, возраст, доза). Шанс = 1 / число повторов'),
      code(r'''
from sklearn.model_selection import cross_val_predict, StratifiedKFold
rows = []
for n, X in reps.items():
    pr = np.zeros(len(meta), int); chance = []
    for key, g in meta.groupby(['type', 'dose']):
        idx = g.index.values; yr = g.replicate.values
        if len(np.unique(yr)) < 2 or np.bincount(yr).max() < 2: continue
        m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
        pr[idx] = cross_val_predict(m, X[idx], yr, cv=StratifiedKFold(3, shuffle=True, random_state=SEED)); chance += [1 / len(np.unique(yr))] * len(idx)
    rows.append(dict(rep=n, replicate_BA=V.balanced_accuracy(meta.replicate.values, pr), chance=np.mean(chance)))
pd.DataFrame(rows).round(3)
'''),
      md('### Кластеризация без учителя и UMAP'),
      code(r'''
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
import umap
prep = lambda n, X: StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(X) if n == 'A_features' else Normalizer().fit_transform(X))
rows = []
for n, X in reps.items():
    Z = PCA(0.95, svd_solver='full').fit_transform(prep(n, X))
    for k in range(2, 7):
        lab = KMeans(k, n_init=20, random_state=SEED).fit_predict(Z)
        rows.append(dict(rep=n, k=k, silhouette=silhouette_score(Z, lab), davies_bouldin=davies_bouldin_score(Z, lab),
                         ARI_dose=adjusted_rand_score(meta.dose, lab), NMI_dose=normalized_mutual_info_score(meta.dose, lab),
                         ARI_species=adjusted_rand_score(meta.species, lab), ARI_replicate=adjusted_rand_score(meta.replicate, lab)))
clus = pd.DataFrame(rows).round(3); clus.to_csv(f'{RES}/clustering.csv', index=False); clus
'''),
      code(r'''
for n in ['A_features', B_NAME]:
    U = umap.UMAP(random_state=SEED).fit_transform(prep(n, reps[n]))
    fig, ax = plt.subplots(1, 5, figsize=(25, 4.5))
    for a, col in zip(ax, ['species', 'age', 'dose', 'culture', 'replicate']):
        for v in sorted(meta[col].unique()):
            mk = (meta[col] == v).values; a.scatter(U[mk, 0], U[mk, 1], s=14, label=str(v))
        a.set_title(f'{n}: {col}'); a.legend(fontsize=6) if col != 'culture' else None
    plt.tight_layout(); fig.savefig(f'{RES}/figs/umap_{n}.png', dpi=90); plt.show()
'''),
      md('### Кривые обучения линейного классификатора: качество в зависимости от числа размеченных культур (представления не обучаются)'),
      code(r'''
def lc_one(frac, r, n, X, kind):
    rng = np.random.default_rng([SEED, int(frac * 100), r]); p = np.zeros(len(y))
    for g in np.unique(groups):
        te = groups == g; tr_c = [c for c in np.unique(groups) if c != g]
        tr = np.isin(groups, rng.choice(tr_c, max(4, int(round(frac * len(tr_c)))), replace=False))
        if len(np.unique(y[tr])) < 2: p[te] = 0.5; continue
        p[te] = make_model(kind).set_params(logisticregression__C=0.1).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return dict(frac=frac, rep=n, run=r, BA=ba(p))
todo = [(f, r, n, X, k) for f in [0.25, 0.5, 0.75, 1.0] for r in range(5 if f < 1 else 1)
        for n, (X, k) in {'A_features': (A, 'features'), B_NAME: (EMB[B_NAME], 'emb'), C_NAME: (EMB[C_NAME], 'emb')}.items()]
lc = pd.DataFrame(Parallel(n_jobs=-1)(delayed(lc_one)(*a) for a in todo)).groupby(['rep', 'frac']).BA.agg(['mean', 'std']).round(3)
lc.to_csv(f'{RES}/learning_curves.csv'); lc
'''),
      ]

C += [md('# Сохранение результатов'),
      code(r'''
# Архив со всеми результатами (таблицы, маски, эмбеддинги, рисунки) -> скачивание на компьютер
import shutil
shutil.make_archive(f'{WORK}/results', 'zip', WORK, 'results')
print('архив:', f'{WORK}/results.zip')
if IN_COLAB:
    from google.colab import files
    files.download(f'{WORK}/results.zip')
'''),
      ]

nb = {'nbformat': 4, 'nbformat_minor': 0,
      'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'A100', 'machine_shape': 'hm'},
                   'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}, 'language_info': {'name': 'python'}},
      'cells': C}
text = json.dumps(nb, ensure_ascii=False, indent=1) + '\n'
if '--check' in sys.argv:
    cur = open(OUT, encoding='utf-8', newline='').read().replace('\r\n', '\n') if os.path.exists(OUT) else ''
    if cur != text:
        sys.exit('notebooks/VKR_colab.ipynb не совпадает с исходниками: запустите python tools/make_colab.py')
    print('ноутбук совпадает с исходниками')
else:
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f: f.write(text)
    print('saved', OUT, len(C), 'cells')
