"""tests/test_refine_interior.py — Refinamiento de la clase interior con DBSCAN."""
import numpy as np

from tests.synthetic_cloud import make_synthetic_interior


def _cluster_de(cluster_labels, truth, clase):
    """id del cluster DBSCAN que mejor cubre la clase-verdad indicada."""
    ids = [c for c in np.unique(cluster_labels) if c != -1]
    return max(ids, key=lambda c: np.sum((cluster_labels == c) & (truth == clase)))


def _params_dummy(centro, r):
    from app.modules.segmentation import FosterParams
    return FosterParams(z_suelo=0.0, cx=centro[0], cy=centro[1], r_tambor_ext=4.4,
                        r_tambor_int=None, z_top_muro=centro[2],
                        centro_cupula=tuple(centro), r_cupula=r)


# ------------------------------------------------------------------ #
# Task 2: dbscan_interior                                             #
# ------------------------------------------------------------------ #
def test_dbscan_separa_el_panel_de_los_demas():
    from app.modules.refine_interior import dbscan_interior
    pts, truth, _, _ = make_synthetic_interior(seed=0)
    cl = dbscan_interior(pts, eps=0.15, min_points=20)
    assert cl.max() >= 3          # varios clusters, no un blob único
    cid = _cluster_de(cl, truth, 7)
    panel = truth == 7
    assert np.mean(cl[panel] == cid) > 0.9


# ------------------------------------------------------------------ #
# Task 3: info_clusters                                               #
# ------------------------------------------------------------------ #
def test_info_clusters_distingue_cascara_y_planaridad():
    from app.modules.refine_interior import dbscan_interior, info_clusters
    pts, truth, centro, r = make_synthetic_interior(seed=0)
    cl = dbscan_interior(pts, eps=0.15, min_points=20)
    infos = info_clusters(pts, cl, centro, r, margen_cascara=0.30)
    by_id = {i.id: i for i in infos}

    panel = by_id[_cluster_de(cl, truth, 7)]
    viga = by_id[_cluster_de(cl, truth, 3)]
    equipo = by_id[_cluster_de(cl, truth, 6)]

    assert panel.frac_en_cascara > 0.8
    assert viga.frac_en_cascara > 0.5
    assert equipo.frac_en_cascara < 0.1
    assert panel.planaridad > viga.planaridad


# ------------------------------------------------------------------ #
# Task 4: clasificar_clusters                                         #
# ------------------------------------------------------------------ #
def test_clasificar_con_id_compuertas():
    from app.modules.refine_interior import (
        COMPUERTAS, CUPULA, DESCARTADO, RefineConfig, clasificar_clusters,
        dbscan_interior, info_clusters)
    pts, truth, centro, r = make_synthetic_interior(seed=0)
    cl = dbscan_interior(pts, eps=0.15, min_points=20)
    infos = info_clusters(pts, cl, centro, r)
    id_panel = _cluster_de(cl, truth, 7)

    mapa = clasificar_clusters(infos, id_compuertas=id_panel, config=RefineConfig())
    assert mapa[id_panel] == COMPUERTAS
    assert mapa[_cluster_de(cl, truth, 3)] == CUPULA
    assert mapa[_cluster_de(cl, truth, 6)] == DESCARTADO


def test_clasificar_auto_propone_compuertas_por_planaridad():
    from app.modules.refine_interior import (
        COMPUERTAS, RefineConfig, clasificar_clusters, dbscan_interior, info_clusters)
    pts, truth, centro, r = make_synthetic_interior(seed=0)
    cl = dbscan_interior(pts, eps=0.15, min_points=20)
    infos = info_clusters(pts, cl, centro, r)
    mapa = clasificar_clusters(infos, id_compuertas=None, config=RefineConfig())
    assert mapa[_cluster_de(cl, truth, 7)] == COMPUERTAS


# ------------------------------------------------------------------ #
# Task 5: refine_interior                                             #
# ------------------------------------------------------------------ #
def test_refine_interior_reetiqueta_y_conserva_otras_clases():
    from app.modules.refine_interior import dbscan_interior, refine_interior
    pts, truth, centro, r = make_synthetic_interior(seed=0)
    labels = np.zeros(len(pts), dtype=int)
    ang = np.linspace(0, 6, 500)
    tambor = np.column_stack([r * np.cos(ang), r * np.sin(ang), np.full(500, -5.0)])
    pts2 = np.vstack([pts, tambor])
    labels2 = np.concatenate([labels, np.full(500, 2)])

    cl = dbscan_interior(pts, eps=0.15, min_points=20)
    id_panel = _cluster_de(cl, truth, 7)

    out = refine_interior(pts2, labels2, _params_dummy(centro, r), id_compuertas=id_panel)
    assert np.all(out[len(pts):] == 2)                       # tambor intacto
    assert np.mean(out[:len(pts)][truth == 7] == 7) > 0.9    # compuertas
    assert np.mean(out[:len(pts)][truth == 3] == 3) > 0.9    # vigas -> cupula
    seg6 = out[:len(pts)][truth == 6]
    # ningún descartado se manda a estructura (cúpula/compuertas); el resto que no
    # es 6 quedó interior(0) por ser ruido de DBSCAN (comportamiento por diseño)
    assert np.mean(np.isin(seg6, [6, 0])) > 0.99
    assert np.mean(seg6 == 6) > 0.85                         # la mayoría se descarta


# ------------------------------------------------------------------ #
# Reetiquetado manual por clusters (elección del usuario)             #
# ------------------------------------------------------------------ #
def test_reetiquetar_por_clusters_manual():
    from app.modules.refine_interior import (
        COMPUERTAS, DESCARTADO, RefineConfig, dbscan_interior,
        reetiquetar_interior_por_clusters)
    pts, truth, centro, r = make_synthetic_interior(seed=0)
    labels = np.zeros(len(pts), dtype=int)
    cl = dbscan_interior(pts, 0.15, 20)
    id_panel = _cluster_de(cl, truth, 7)
    id_tele = _cluster_de(cl, truth, 6)

    out = reetiquetar_interior_por_clusters(
        pts, labels, RefineConfig(), id_compuertas=id_panel, ids_descartar=[id_tele])
    assert np.mean(out[truth == 7] == COMPUERTAS) > 0.9   # panel -> compuertas
    assert (out[cl == id_tele] == DESCARTADO).all()        # cluster marcado -> descartado
    assert np.mean(out[truth == 3] == 0) > 0.9             # vigas no marcadas siguen interior


def test_reetiquetar_marca_suelo():
    from app.modules.refine_interior import (
        SUELO, RefineConfig, dbscan_interior, reetiquetar_interior_por_clusters)
    pts, truth, centro, r = make_synthetic_interior(seed=0)
    labels = np.zeros(len(pts), dtype=int)
    cl = dbscan_interior(pts, 0.15, 20)
    id_tele = _cluster_de(cl, truth, 6)
    out = reetiquetar_interior_por_clusters(pts, labels, RefineConfig(), ids_suelo=[id_tele])
    assert (out[cl == id_tele] == SUELO).all()             # cluster marcado -> suelo(1)


# ------------------------------------------------------------------ #
# Task 6: eliminar_descartados                                        #
# ------------------------------------------------------------------ #
def test_eliminar_descartados_quita_solo_clase_6():
    from app.modules.refine_interior import eliminar_descartados
    labels = np.array([0, 3, 6, 7, 6, 2, 6])
    pts = np.arange(len(labels) * 3, dtype=float).reshape(-1, 3)
    p2, l2 = eliminar_descartados(pts, labels)
    assert (l2 == 6).sum() == 0
    assert len(l2) == 4
    assert set(l2.tolist()) == {0, 3, 7, 2}
    assert len(p2) == len(l2)
