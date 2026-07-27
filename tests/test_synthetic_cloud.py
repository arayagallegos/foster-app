"""tests/test_synthetic_cloud.py — La nube sintética enriquecida."""
import numpy as np

from tests.synthetic_cloud import (
    H_MURO, R_TAMBOR, make_synthetic_foster, make_synthetic_foster_dificil,
)


def test_sintetica_6clases_tiene_todas_las_clases_y_dimensiones():
    from tests.synthetic_cloud import make_synthetic_foster_6clases
    pts, lab = make_synthetic_foster_6clases(seed=0)
    assert set(np.unique(lab)) == {0, 1, 2, 3, 4, 5}
    # radio del tambor (clase 2) ≈ 4.41
    tam = pts[lab == 2]
    r = np.hypot(tam[:, 0], tam[:, 1])
    assert 4.3 < np.median(r) < 4.5
    # cornisa (clase 4) sobresale (radio máx)
    cor = pts[lab == 4]
    assert np.hypot(cor[:, 0], cor[:, 1]).max() > 4.6
    # contrafuertes (clase 5) fuera del tambor
    con = pts[lab == 5]
    assert np.hypot(con[:, 0], con[:, 1]).max() > 4.41


def test_sintetica_6clases_verdad_bien_planteada():
    from scipy.spatial import cKDTree

    from tests.synthetic_cloud import make_synthetic_foster_6clases
    pts, lab = make_synthetic_foster_6clases(seed=1)
    # ningún par de clases distintas en (casi) la misma posición
    tree = cKDTree(pts)
    pares = tree.query_pairs(r=0.01, output_type="ndarray")
    if len(pares):
        assert np.all(lab[pares[:, 0]] == lab[pares[:, 1]])


def test_defaults_igual_que_nube_simple():
    # sin activar nada, la composición base sigue siendo suelo/tambor/cupula(+outliers)
    pts, labels = make_synthetic_foster(seed=0)
    assert set(np.unique(labels)) == {0, 1, 2, 3}
    suelo = pts[labels == 1]
    assert np.allclose(suelo[:, 2], 0.0, atol=0.05)     # suelo plano en z=0


def test_pendiente_suelo_inclina_el_plano():
    pts, labels = make_synthetic_foster(seed=1, pendiente_suelo_deg=5.0,
                                        outlier_frac=0.0)
    suelo = pts[labels == 1]
    # con 5° de pendiente, z del suelo ya no es constante
    assert suelo[:, 2].std() > 0.05
    # la pendiente esperada ~ tan(5°) * radio
    assert suelo[:, 2].max() - suelo[:, 2].min() > 0.3


def test_contrafuertes_e_interior_son_resto():
    pts, labels = make_synthetic_foster(seed=2, n_contrafuertes=4, n_interior=500,
                                        outlier_frac=0.0)
    # los puntos añadidos tienen etiqueta 0 (resto)
    assert (labels == 0).sum() >= 500


def test_junta_realista_tiene_falda_bajo_el_ecuador():
    pts, labels = make_synthetic_foster(seed=3, junta_realista=True, outlier_frac=0.0)
    cupula = pts[labels == 3]
    # el mecanismo clave: hay puntos de cúpula POR DEBAJO del ecuador (z < H_MURO),
    # que es donde el corte rígido z=sz los misclasifica como tambor
    falda = cupula[cupula[:, 2] < H_MURO]
    assert len(falda) > 0
    # y sobresale levemente del tambor (overhang)
    r_base = np.hypot(cupula[:, 0], cupula[:, 1])
    assert r_base.max() > R_TAMBOR


def test_dificil_activa_todo_y_es_reproducible():
    a_pts, a_lab = make_synthetic_foster_dificil(seed=7)
    b_pts, b_lab = make_synthetic_foster_dificil(seed=7)
    assert np.array_equal(a_pts, b_pts) and np.array_equal(a_lab, b_lab)
    assert len(a_pts) > 20_000


def test_sintetico_interior_tiene_partes_y_geometria():
    from tests.synthetic_cloud import make_synthetic_interior
    pts, truth, centro, r = make_synthetic_interior(seed=0)
    centro = np.asarray(centro)
    assert set(np.unique(truth)) == {3, 6, 7}
    d = np.linalg.norm(pts - centro, axis=1)
    # compuertas(7) y vigas(3) están sobre la cáscara (|d - r| pequeño)
    for clase in (7, 3):
        assert np.median(np.abs(d[truth == clase] - r)) < 0.15
    # descartado(6) está lejos de la cáscara
    assert np.median(np.abs(d[truth == 6] - r)) > 1.0
