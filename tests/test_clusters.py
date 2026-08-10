"""tests/test_clusters.py — Agrupación por densidad genérica."""
import numpy as np
import pytest

from app.modules.clusters import (
    RUIDO, InfoCluster, clusterizar, espaciado_medio, info_clusters,
    marcar_pequenos_como_ruido, mascara_de,
)


def _tres_bloques(seed=0):
    """Tres nubes bien separadas: 2000, 500 y 60 puntos."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.10, (2000, 3))
    b = rng.normal(0, 0.10, (500, 3)) + np.array([10.0, 0, 0])
    c = rng.normal(0, 0.10, (60, 3)) + np.array([0, 10.0, 0])
    return np.vstack([a, b, c])


def test_clusterizar_separa_grupos_alejados():
    pts = _tres_bloques()
    et = clusterizar(pts, eps=0.30, min_points=10)
    assert len({c for c in np.unique(et) if c != RUIDO}) == 3
    # cada bloque cae entero en un mismo cluster
    for ini, fin in ((0, 2000), (2000, 2500), (2500, 2560)):
        assert len(set(et[ini:fin])) == 1


def test_clusterizar_devuelve_una_etiqueta_por_punto_aunque_submuestree():
    """La submuestra es la red de seguridad para nubes enormes, pero el
    resultado debe cubrir TODOS los puntos, no solo los muestreados."""
    pts = _tres_bloques()
    et = clusterizar(pts, eps=0.30, min_points=10, max_puntos=800)
    assert len(et) == len(pts)
    assert (et != RUIDO).sum() > 0.9 * len(pts)


def test_la_submuestra_no_cambia_los_grupos():
    pts = _tres_bloques()
    completo = clusterizar(pts, eps=0.30, min_points=10)
    reducido = clusterizar(pts, eps=0.30, min_points=10, max_puntos=800)
    # los ids pueden diferir; lo que debe coincidir es la PARTICIÓN
    for a, b in ((0, 2000), (2000, 2500)):
        assert len(set(reducido[a:b])) == 1
    assert len({c for c in np.unique(reducido) if c != RUIDO}) == \
           len({c for c in np.unique(completo) if c != RUIDO})


def test_clusterizar_marca_ruido_lo_disperso():
    rng = np.random.default_rng(1)
    denso = rng.normal(0, 0.05, (1000, 3))
    disperso = rng.uniform(-20, 20, (50, 3))
    et = clusterizar(np.vstack([denso, disperso]), eps=0.2, min_points=20)
    assert (et[1000:] == RUIDO).sum() >= 40


def test_clusterizar_con_nube_vacia():
    assert len(clusterizar(np.empty((0, 3)))) == 0


def test_info_clusters_ordena_de_mayor_a_menor():
    pts = _tres_bloques()
    et = clusterizar(pts, eps=0.30, min_points=10)
    infos = info_clusters(pts, et)
    assert [i.n_pts for i in infos] == sorted([i.n_pts for i in infos], reverse=True)
    assert all(isinstance(i, InfoCluster) for i in infos)
    assert infos[0].n_pts == 2000


def test_la_planaridad_distingue_una_superficie_de_una_linea():
    """Es lo que separa un panel de un enrejado o un cable."""
    rng = np.random.default_rng(2)
    plano = np.column_stack([rng.uniform(-1, 1, (800, 2)),
                             rng.normal(0, 0.005, 800)])
    linea = np.column_stack([rng.uniform(-1, 1, 800),
                             rng.normal(0, 0.005, 800),
                             rng.normal(0, 0.005, 800)])
    pts = np.vstack([plano, linea + np.array([20.0, 0, 0])])
    et = np.array([0] * 800 + [1] * 800)
    por_id = {i.id: i for i in info_clusters(pts, et)}
    assert por_id[0].planaridad > 0.8      # superficie
    assert por_id[1].planaridad < 0.2      # lineal


def test_el_umbral_marca_como_ruido_los_clusters_chicos():
    """El umbral se aplica DESPUÉS de DBSCAN, así que moverlo no obliga a
    repetir la operación cara."""
    et = np.array([0] * 100 + [1] * 30 + [2] * 5 + [RUIDO] * 10)
    filtrado = marcar_pequenos_como_ruido(et, min_pts=50)
    assert set(np.unique(filtrado)) == {RUIDO, 0}
    assert (filtrado == RUIDO).sum() == 45


def test_el_umbral_no_toca_el_ruido_previo():
    et = np.array([0] * 100 + [RUIDO] * 10)
    assert (marcar_pequenos_como_ruido(et, min_pts=1) == RUIDO).sum() == 10


def test_mascara_de_selecciona_los_clusters_pedidos():
    et = np.array([0, 1, 2, 1, RUIDO, 0])
    assert list(mascara_de(et, [0, 2])) == [True, False, True, False, False, True]
    assert not mascara_de(et, []).any()


def test_espaciado_medio_es_la_referencia_para_eps():
    """Con una rejilla de paso conocido, el espaciado medido debe ser ese paso."""
    g = np.arange(0, 2.0, 0.10)
    x, y = np.meshgrid(g, g)
    pts = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    assert espaciado_medio(pts) == pytest.approx(0.10, rel=0.05)


def test_el_espaciado_efectivo_es_el_de_la_submuestra_no_el_de_la_nube():
    """Es el bug que hacía sugerir un eps 2.6x demasiado chico: DBSCAN no ve la
    nube completa, ve la submuestra, y ahí los puntos están más separados."""
    from app.modules.clusters import espaciado_efectivo

    rng = np.random.default_rng(0)
    # puntos sobre una superficie: el espaciado crece con sqrt del diezmado
    n = 40_000
    pts = np.column_stack([rng.uniform(0, 1, (n, 2)), np.zeros(n)])
    completo = espaciado_medio(pts)
    efectivo = espaciado_efectivo(pts, max_puntos=n // 9)

    assert efectivo > completo
    assert efectivo / completo == pytest.approx(3.0, rel=0.15)   # sqrt(9)


def test_el_espaciado_efectivo_coincide_si_no_hay_submuestreo():
    from app.modules.clusters import espaciado_efectivo
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 1, (3000, 3))
    assert espaciado_efectivo(pts, max_puntos=10_000) == pytest.approx(
        espaciado_medio(pts))


def test_vecinos_tipicos_es_el_techo_de_min_points():
    """Si min_points supera este número, ningún punto llega a ser núcleo y TODA
    la nube queda como ruido."""
    from app.modules.clusters import vecinos_tipicos

    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(0, 10, (50_000, 2)), np.zeros(50_000)])
    s = espaciado_medio(pts)
    v = vecinos_tipicos(pts, 3 * s)

    # con min_points por debajo del conteo hay clusters; por encima, solo ruido
    assert (clusterizar(pts, eps=3 * s, min_points=max(2, v // 2)) != RUIDO).any()
    assert not (clusterizar(pts, eps=3 * s, min_points=v * 5) != RUIDO).any()


def test_las_proporciones_reproducen_la_corrida_buena_del_interior():
    """Calibración: la corrida documentada usó eps=0.15 y min_points=20 sobre
    una nube de espaciado 0.0215 m donde un punto tenía 85 vecinos. Las
    constantes deben devolver esos valores."""
    from app.modules.clusters import EPS_POR_ESPACIADO, MIN_POINTS_POR_VECINOS

    assert EPS_POR_ESPACIADO * 0.0215 == pytest.approx(0.15, abs=0.01)
    assert 85 * MIN_POINTS_POR_VECINOS == pytest.approx(20, abs=1)


def test_la_proporcion_se_traslada_a_otra_densidad():
    """El valor absoluto NO se traslada: 0.15 m sobre una nube 3.5x más densa
    funde toda la estructura en un cluster. La proporción sí."""
    from app.modules.clusters import EPS_POR_ESPACIADO

    rng = np.random.default_rng(0)
    # misma superficie, dos densidades distintas
    def cascara(n):
        th = rng.uniform(0, 2 * np.pi, n); ph = rng.uniform(0.2, np.pi / 2, n)
        r = 5.0 + rng.normal(0, 0.005, n)
        return np.column_stack([r * np.sin(ph) * np.cos(th),
                                r * np.sin(ph) * np.sin(th), r * np.cos(ph)])

    for n in (60_000, 250_000):
        pts = cascara(n)
        s = espaciado_medio(pts)
        et = clusterizar(pts, eps=EPS_POR_ESPACIADO * s, min_points=10)
        agrupados = (et != RUIDO).mean()
        assert agrupados > 0.9, f"con {n} puntos solo se agrupó {agrupados:.0%}"
