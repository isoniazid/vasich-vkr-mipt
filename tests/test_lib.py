"""Быстрые проверки vkr_lib на синтетических изображениях (без данных и без GPU)."""
import os, sys
import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import vkr_lib as V  # noqa: E402

PX = V.TARGET_PX


def synthetic_cell(two_nuclei=False, seed=0):
    """Изображение 3 x 600 x 600 (lyso, mito, nuc) с одной или двумя клетками."""
    rng = np.random.default_rng(seed)
    H = W = 600
    yy, xx = np.mgrid[:H, :W]
    centers = [(300, 300)] + ([(300, 520)] if two_nuclei else [])
    nuc = np.zeros((H, W)); cell = np.zeros((H, W), bool)
    for cy, cx in centers:
        nuc += 200 * (((yy - cy) / 70) ** 2 + ((xx - cx) / 55) ** 2 < 1)        # ядро ~ 25 x 20 мкм
        cell |= ((yy - cy) / 220) ** 2 + ((xx - cx) / 160) ** 2 < 1
    mito = np.zeros((H, W))
    for _ in range(300):                                                        # короткие «нити»
        y0, x0 = rng.integers(0, H), rng.integers(0, W)
        ang = rng.uniform(0, np.pi); L = rng.integers(5, 25)
        ys = (y0 + np.arange(L) * np.sin(ang)).astype(int).clip(0, H - 1)
        xs = (x0 + np.arange(L) * np.cos(ang)).astype(int).clip(0, W - 1)
        mito[ys, xs] = 220
    mito = ndi.gaussian_filter(mito, 1) * cell * 3
    lyso = np.zeros((H, W))
    pts = rng.integers(0, H, (400, 2)); lyso[pts[:, 0], pts[:, 1]] = 255
    lyso = ndi.gaussian_filter(lyso, 1.5) * 20 * cell
    arr = np.stack([lyso, mito, ndi.gaussian_filter(nuc, 2)]).clip(0, 255).astype(np.float32)
    return arr


def test_nuclei_not_split():
    arr = synthetic_cell()
    lab = V.seg_nuclei_otsu(arr[2], PX)
    assert lab.max() == 1, 'одиночное овальное ядро не должно делиться водоразделом'


def test_two_nuclei_two_territories():
    arr = synthetic_cell(two_nuclei=True)
    s = V.segment_all(arr, PX)
    assert s['nuc_lab'].max() == 2
    assert s['nuc_central'] == s['nuc_lab'][300, 300], 'центральная клетка — та, чье ядро ближе к центру кадра'
    assert not s['cell'][300, 560], 'соседняя клетка не входит в анализируемую'


def test_features_have_no_intensity_and_qc_prefix():
    arr = synthetic_cell()
    f = V.extract_features(arr, PX, V.segment_all(arr, PX), n_null=20)
    assert f['qc_nuc_found'] == 1 and f['qc_nuc_touches_border'] == 0
    assert 330 < f['nuc_area'] < 460   # эллипс 70 x 55 пкс при 0,18 мкм/пкс ≈ 392 мкм²
    assert not any('intensity' in k or 'mean_int' in k for k in f)
    assert 'mito_area_frac' in f and 'lyso_near_mito_Z' in f


def test_tiles_isolate_zeroes_outside_cell():
    arr = synthetic_cell(two_nuclei=True)
    s = V.segment_all(arr, PX)
    t0, w0 = V.tiles_for_net(arr, s['cell'], isolate=False)
    t1, w1 = V.tiles_for_net(arr, s['cell'], isolate=True)
    assert t0.shape == t1.shape and np.allclose(w0, w1)
    assert t1.sum() < t0.sum()


def test_focus_measure_is_brightness_invariant():
    arr = synthetic_cell()
    ch = arr[1]
    assert V.focus_measure(ch) == pytest.approx(V.focus_measure(ch * 0.3), rel=0.05)
    assert V.focus_measure(ndi.gaussian_filter(ch, 3)) < V.focus_measure(ch)


def test_stats_helpers():
    y = np.array([0, 0, 1, 1, 1, 1])
    assert V.balanced_accuracy(y, y) == 1.0
    assert V.balanced_accuracy(y, 1 - y) == 0.0
    adj = V.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    blocks = np.array(['a', 'a', 'b', 'b', 'b', 'b'])
    p = V.permute_within(y, blocks, np.random.default_rng(0))
    assert sorted(p[:2]) == [0, 0] and p[2:].sum() == 4
    X = np.random.default_rng(0).normal(size=(50, 5))
    assert V.linear_cka(X, X) == pytest.approx(1.0)


def test_hier_bootstrap_keeps_dose_ratio():
    rows = [dict(type=t, culture=f'{t}_c{c}', dose=d) for t in ['A', 'B'] for c in range(3) for d in [0, 5, 10]]
    meta = pd.DataFrame(rows)
    for idx in V.hier_bootstrap(meta, B=20, seed=1):
        d = meta.dose.values[idx]
        assert (d == 0).sum() == (d == 5).sum() == (d == 10).sum() == 6
