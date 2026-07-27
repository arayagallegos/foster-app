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


def test_perfil_radios_recupera_cilindro():
    from app.modules.segmentation import perfil_radios
    rng = np.random.default_rng(0)
    # cilindro r=2.0, z en [0,3]
    th = rng.uniform(0, 2 * np.pi, 20000)
    z = rng.uniform(0, 3, 20000)
    pts = np.column_stack([2.0 * np.cos(th), 2.0 * np.sin(th), z])
    pts += rng.normal(0, 0.01, pts.shape)
    bandas = perfil_radios(pts, paso=0.2)
    assert len(bandas) >= 10
    radios = [b.r_ext for b in bandas]
    assert np.allclose(radios, 2.0, atol=0.05)   # radio constante recuperado


def test_segment_by_profile_clasifica_6_clases():
    from app.modules.segmentation import segment_by_profile
    from app.modules.segmentation_metrics import evaluate
    from tests.synthetic_cloud import make_synthetic_foster_6clases
    pts, truth = make_synthetic_foster_6clases(seed=2)
    res = segment_by_profile(pts)
    assert set(np.unique(res.labels)).issubset({0, 1, 2, 3, 4, 5})
    m = evaluate(res.labels, truth, n_classes=6)
    assert m.per_class[2].iou > 0.6      # tambor
    assert m.per_class[3].iou > 0.6      # cúpula
    assert m.per_class[4].iou > 0.4      # cornisa
    assert m.per_class[5].iou > 0.4      # contrafuerte


def test_segment_by_profile_es_determinista():
    """Regresión: sin semilla en el RANSAC del suelo, la segmentación variaba entre
    corridas (rompía el flujo de elegir ids de cluster y reejecutar)."""
    from app.modules.segmentation import segment_by_profile
    from tests.synthetic_cloud import make_synthetic_foster_6clases
    pts, _ = make_synthetic_foster_6clases(seed=3)
    a = segment_by_profile(pts).labels
    b = segment_by_profile(pts).labels
    assert np.array_equal(a, b)


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
    for clase in ("interior", "suelo", "tambor", "cupula"):
        assert rutas[clase].exists()
    data = json.loads((tmp_path / "parametros_foster.json").read_text())
    assert data["r_tambor_ext"] == pytest.approx(2.5, abs=0.02)
    assert data["r_tambor_int"] is None


def test_save_segments_soporta_6_clases(tmp_path):
    import open3d as o3d
    from app.modules.segmentation import (
        FosterParams, SegmentationResult, save_segments)
    pts = np.random.default_rng(0).uniform(0, 5, (600, 3))
    labels = np.arange(600) % 6            # las 6 clases presentes
    params = FosterParams(z_suelo=0.0, cx=0.0, cy=0.0, r_tambor_ext=2.5,
                          r_tambor_int=None, z_top_muro=3.0,
                          centro_cupula=(0.0, 0.0, 3.0), r_cupula=2.5)
    rutas = save_segments(pts, SegmentationResult(labels=labels, params=params), tmp_path)
    for nombre in ("interior", "suelo", "tambor", "cupula", "cornisa", "contrafuerte"):
        assert nombre in rutas


def test_save_segments_soporta_8_clases(tmp_path):
    from app.modules.segmentation import (
        FosterParams, SegmentationResult, save_segments)
    pts = np.random.default_rng(0).uniform(0, 5, (800, 3))
    labels = np.arange(800) % 8            # las 8 clases presentes
    params = FosterParams(z_suelo=0.0, cx=0.0, cy=0.0, r_tambor_ext=2.5,
                          r_tambor_int=None, z_top_muro=3.0,
                          centro_cupula=(0.0, 0.0, 3.0), r_cupula=2.5)
    rutas = save_segments(pts, SegmentationResult(labels=labels, params=params), tmp_path)
    for nombre in ("interior", "suelo", "tambor", "cupula", "cornisa",
                   "contrafuerte", "descartado", "compuertas"):
        assert nombre in rutas and rutas[nombre].exists()


def test_save_segments_escribe_nube_completa_coloreada(tmp_path):
    """Además de los archivos por clase, un único .ply con TODOS los puntos
    coloreados según su clase, para inspeccionar el modelo completo en el visor."""
    import open3d as o3d

    from app.modules.segmentation import save_segments

    pts, _ = make_synthetic_foster(seed=9)
    res = segment_foster(pts)
    rutas = save_segments(pts, res, tmp_path)

    assert "completo" in rutas
    completo = rutas["completo"]
    assert completo.exists()

    pcd = o3d.io.read_point_cloud(str(completo))
    # No se pierde ni se duplica ningún punto
    assert len(pcd.points) == len(pts)
    # Está coloreado (por clase), no monocromo
    assert pcd.has_colors()
    colores_unicos = np.unique(np.asarray(pcd.colors), axis=0)
    assert len(colores_unicos) >= 2   # al menos dos clases presentes


# ------------------------------------------------------------------ #
# Fase 1: línea base sobre la nube difícil (documenta el problema)     #
# ------------------------------------------------------------------ #

def test_linea_base_dificil_tiene_confusion_cupula_tambor():
    from app.modules.segmentation import segment_foster
    from app.modules.segmentation_metrics import evaluate
    from tests.synthetic_cloud import make_synthetic_foster_dificil

    pts, truth = make_synthetic_foster_dificil(seed=11)
    res = segment_foster(pts)
    m = evaluate(res.labels, truth, n_classes=4)
    # Documenta el problema: en la línea base, cúpula(3) y tambor(2) se confunden
    # -> hay puntos cuya verdad es cúpula pero se predicen tambor (confusion[3,2]>0).
    assert m.confusion[3, 2] > 0
    # macro IoU de partida está por debajo de lo perfecto (hay margen que mejorar)
    assert m.macro_iou < 0.95
