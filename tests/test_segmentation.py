"""tests/test_segmentation.py — Tests del módulo 4 (segmentación RANSAC)."""
import numpy as np
import pytest

from app.modules.segmentation import (
    fit_circle_ransac,
    fit_sphere_ransac,
    segment_foster,
    segment_ground,
)
from tests.synthetic_cloud import H_MURO, R_TAMBOR, _dome, make_synthetic_foster


def test_generador_sintetico_produce_clases_y_rangos():
    pts, labels = make_synthetic_foster(seed=1)
    assert pts.shape[1] == 3 and len(pts) == len(labels)
    assert set(np.unique(labels)) == {0, 1, 2, 3}
    tambor = pts[labels == 2]
    r = np.hypot(tambor[:, 0], tambor[:, 1])
    assert np.median(r) == pytest.approx(R_TAMBOR, abs=0.02)
    cupula = pts[labels == 3]
    assert cupula[:, 2].min() > H_MURO - 0.1


# ------------------------------------------------------------------ #
# Task 2: círculo RANSAC 2D                                            #
# ------------------------------------------------------------------ #

def test_fit_circle_ransac_recupera_con_outliers():
    rng = np.random.default_rng(3)
    theta = rng.uniform(0, 2 * np.pi, 2000)
    ring = np.column_stack([1.0 + 2.5 * np.cos(theta), -2.0 + 2.5 * np.sin(theta)])
    ring += rng.normal(0, 0.01, ring.shape)
    outliers = rng.uniform(-6, 6, (900, 2))
    xy = np.vstack([ring, outliers])

    cx, cy, r, mask = fit_circle_ransac(xy, eps=0.05, seed=0)
    assert (cx, cy, r) == pytest.approx((1.0, -2.0, 2.5), abs=0.02)
    assert mask[:2000].mean() > 0.95      # casi todo el anillo es inlier
    assert mask[2000:].mean() < 0.10      # casi ningún outlier


def test_fit_circle_ransac_sin_circulo_lanza_error():
    rng = np.random.default_rng(4)
    xy = rng.uniform(-6, 6, (500, 2))
    with pytest.raises(RuntimeError, match="c[ií]rculo"):
        fit_circle_ransac(xy, eps=0.01, n_iters=200, seed=0)


# ------------------------------------------------------------------ #
# Task 3: esfera RANSAC                                                #
# ------------------------------------------------------------------ #

def test_fit_sphere_ransac_recupera_con_outliers():
    rng = np.random.default_rng(5)
    dome = _dome(2.5, 3.0, 3000, rng) + np.array([0.5, 0.3, 0.0])
    dome += rng.normal(0, 0.01, dome.shape)
    outliers = np.column_stack([rng.uniform(-6, 6, 1200),
                                rng.uniform(-6, 6, 1200),
                                rng.uniform(0, 7, 1200)])
    pts = np.vstack([dome, outliers])

    centro, r, mask = fit_sphere_ransac(pts, eps=0.08, seed=0)
    assert centro == pytest.approx([0.5, 0.3, 3.0], abs=0.03)
    assert r == pytest.approx(2.5, abs=0.03)
    assert mask[:3000].mean() > 0.95


# ------------------------------------------------------------------ #
# Task 4: plano de suelo                                               #
# ------------------------------------------------------------------ #

def test_segment_ground_encuentra_cota_cero():
    pts, labels = make_synthetic_foster(seed=2)
    z_suelo, mask = segment_ground(pts)
    assert z_suelo == pytest.approx(0.0, abs=0.02)
    # la mayoría del suelo verdadero queda en la máscara
    assert mask[labels == 1].mean() > 0.90


# ------------------------------------------------------------------ #
# Task 5: orquestador segment_foster                                   #
# ------------------------------------------------------------------ #

def test_segment_foster_etiquetas_y_parametros():
    pts, truth = make_synthetic_foster(seed=6)
    res = segment_foster(pts)
    p = res.params
    assert p.z_suelo == pytest.approx(0.0, abs=0.02)
    assert (p.cx, p.cy) == pytest.approx((0.0, 0.0), abs=0.02)
    assert p.r_tambor_ext == pytest.approx(2.5, abs=0.02)
    assert p.r_tambor_int is None
    assert p.r_cupula == pytest.approx(2.5, abs=0.03)
    assert p.z_top_muro == pytest.approx(3.0, abs=0.05)
    # exactitud por clase > 90 % (sobre las etiquetas verdaderas no-outlier)
    for clase in (1, 2, 3):
        acc = (res.labels[truth == clase] == clase).mean()
        assert acc > 0.90, f"clase {clase}: {acc:.2%}"


def test_segment_foster_detecta_cara_interior_del_muro():
    pts, _ = make_synthetic_foster(seed=7, r_int=2.2)
    p = segment_foster(pts).params
    assert p.r_tambor_int == pytest.approx(2.2, abs=0.02)
    assert p.r_tambor_ext - p.r_tambor_int == pytest.approx(0.3, abs=0.02)


# ------------------------------------------------------------------ #
# Task 6: salidas                                                      #
# ------------------------------------------------------------------ #

def test_save_segments_escribe_ply_y_json(tmp_path):
    import json

    from app.modules.segmentation import save_segments

    pts, _ = make_synthetic_foster(seed=8)
    res = segment_foster(pts)
    rutas = save_segments(pts, res, tmp_path)
    for clase in ("resto", "suelo", "tambor", "cupula"):
        assert rutas[clase].exists()
    data = json.loads((tmp_path / "parametros_foster.json").read_text())
    assert data["r_tambor_ext"] == pytest.approx(2.5, abs=0.02)
    assert data["r_tambor_int"] is None
