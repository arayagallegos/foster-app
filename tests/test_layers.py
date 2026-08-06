"""tests/test_layers.py — Modelo de capas del recorte iterativo."""
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from app.core.layers import CloudLayer, LayerStack


def _pcd(n=100, seed=0, con_color=True):
    rng = np.random.default_rng(seed)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rng.uniform(0, 10, (n, 3))))
    if con_color:
        pcd.colors = o3d.utility.Vector3dVector(rng.uniform(0, 1, (n, 3)))
    return pcd


def _stack_con_split(n=100, n_keep=60):
    stack = LayerStack()
    stack.reset(_pcd(n))
    keep = np.zeros(n, dtype=bool)
    keep[:n_keep] = True
    recorte, descarte = stack.split_active(keep)
    return stack, recorte, descarte


def test_split_active_fino_recorta_la_nube_fina(tmp_path):
    # capa-scan: gruesa en memoria + fina en disco (más puntos)
    rng = np.random.default_rng(0)
    fino = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(rng.uniform(0, 10, (4000, 3))))
    fp = tmp_path / "scan_00.ply"
    o3d.io.write_point_cloud(str(fp), fino)
    grueso = fino.voxel_down_sample(0.5)

    stack = LayerStack()
    stack.layers = [CloudLayer(name="Scan 00", pcd=grueso, fine_path=fp)]
    stack.active_index = 0

    keep_grueso = np.asarray(grueso.points)[:, 0] < 5.0
    def fine_keep_fn(p):
        return np.asarray(p.points)[:, 0] < 5.0   # mismo criterio (caja)

    recorte, descarte = stack.split_active_fino(keep_grueso, fine_keep_fn, tmp_path)
    assert recorte.fine_path is not None and recorte.fine_path.exists()
    assert descarte.fine_path is not None and descarte.fine_path.exists()
    # el fino del recorte tiene solo los x<5 de los 4000 (≈ la mitad, no del grueso)
    recorte_fino = o3d.io.read_point_cloud(str(recorte.fine_path))
    assert len(recorte_fino.points) > len(recorte.pcd.points)   # fino > grueso
    assert np.all(np.asarray(recorte_fino.points)[:, 0] < 5.0)


def test_split_active_fino_sin_fine_path_equivale_a_split(tmp_path):
    stack = LayerStack()
    stack.reset(_pcd(100))
    keep = np.zeros(100, dtype=bool)
    keep[:60] = True
    recorte, descarte = stack.split_active_fino(keep, lambda p: None, tmp_path)
    assert recorte.fine_path is None and descarte.fine_path is None
    assert len(recorte.pcd.points) == 60


def test_merge_visible_fine_usa_los_ply_de_disco(tmp_path):
    def _mk(idx, n_fino):
        rng = np.random.default_rng(idx)
        fino = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(rng.uniform(0, 5, (n_fino, 3))))
        fp = tmp_path / f"scan_{idx:02d}.ply"
        o3d.io.write_point_cloud(str(fp), fino)
        grueso = fino.voxel_down_sample(0.5)   # menos puntos
        return CloudLayer(name=f"Scan {idx}", pcd=grueso, fine_path=fp)

    stack = LayerStack()
    stack.layers = [_mk(0, 4000), _mk(1, 4000)]
    stack.active_index = 0
    stack.layers[1].visible = False            # apagar el scan 1

    merged = stack.merge_visible_fine()
    # usa el FINO del scan 0 (4000), no el grueso en memoria
    assert len(merged.points) > len(stack.layers[0].pcd.points)
    assert 3000 < len(merged.points) <= 4000   # solo el scan 0 visible


def test_merge_visible_fine_sin_fine_path_usa_memoria():
    stack = LayerStack()
    stack.reset(_pcd(100))                      # capa normal, sin fine_path
    merged = stack.merge_visible_fine()
    assert len(merged.points) == 100


def test_cloudlayer_acepta_fine_path(tmp_path):
    p = tmp_path / "scan_00.ply"
    capa = CloudLayer(name="Scan 00", pcd=_pcd(10), fine_path=p)
    assert capa.fine_path == p
    # default None
    normal = CloudLayer(name="x", pcd=_pcd(10))
    assert normal.fine_path is None


def test_split_active_hijas_sin_fine_path():
    stack = LayerStack()
    stack.reset(_pcd(100))
    stack.layers[0] = CloudLayer(name="Scan 00", pcd=stack.layers[0].pcd,
                                 fine_path=Path("x.ply"))
    keep = np.zeros(100, dtype=bool)
    keep[:60] = True
    recorte, descarte = stack.split_active(keep)
    assert recorte.fine_path is None and descarte.fine_path is None


def test_reset_crea_capa_original_activa():
    stack = LayerStack()
    stack.reset(_pcd())
    assert len(stack) == 1
    assert stack.active.name == "Original"
    assert stack.active.visible


def test_split_active_no_destructivo():
    stack, recorte, descarte = _stack_con_split()

    # La fuente queda intacta pero oculta; recorte activo y visible; descarte oculto
    assert len(stack) == 3
    original = stack.layers[0]
    assert original.name == "Original"
    assert len(original.pcd.points) == 100
    assert not original.visible

    assert stack.active is recorte
    assert recorte.name == "Recorte 1" and recorte.visible
    assert len(recorte.pcd.points) == 60
    assert descarte.name == "Descarte 1" and not descarte.visible
    assert len(descarte.pcd.points) == 40
    assert recorte.pcd.has_colors() and descarte.pcd.has_colors()


def test_split_active_mascara_invalida_lanza_error():
    stack = LayerStack()
    stack.reset(_pcd(50))
    with pytest.raises(ValueError):
        stack.split_active(np.ones(50, dtype=bool))       # nada que separar
    with pytest.raises(ValueError):
        stack.split_active(np.zeros(50, dtype=bool))      # recorte vacío
    with pytest.raises(ValueError):
        stack.split_active(np.ones(10, dtype=bool))       # largo incorrecto


def test_remove_unica_capa_lanza_error():
    stack = LayerStack()
    stack.reset(_pcd())
    with pytest.raises(ValueError):
        stack.remove(0)


def test_remove_capa_activa_reasigna_activa():
    stack, _, _ = _stack_con_split()
    assert stack.active_index == 1
    stack.remove(1)
    assert len(stack) == 2
    assert stack.active_index == 0


def test_merge_visible_concatena_visibles():
    stack, _, _ = _stack_con_split()
    # Solo "Recorte 1" está visible tras el split
    assert len(stack.merge_visible().points) == 60
    stack.set_visible(2, True)   # mostrar el descarte
    assert len(stack.merge_visible().points) == 100


def test_merge_sin_visibles_lanza_error():
    stack = LayerStack()
    stack.reset(_pcd())
    stack.set_visible(0, False)
    with pytest.raises(ValueError):
        stack.merge_visible()


def test_insert_layer_restaura_capa_eliminada():
    stack, _, descarte = _stack_con_split()
    stack.remove(2)
    assert len(stack) == 2

    stack.insert_layer(2, descarte)
    assert len(stack) == 3
    assert stack.layers[2].name == "Descarte 1"
    assert stack.active.name == "Recorte 1"


def test_nombres_de_split_incrementales():
    stack, recorte1, descarte1 = _stack_con_split()
    keep2 = np.zeros(60, dtype=bool)
    keep2[:30] = True
    recorte2, descarte2 = stack.split_active(keep2)
    assert (recorte1.name, descarte1.name) == ("Recorte 1", "Descarte 1")
    assert (recorte2.name, descarte2.name) == ("Recorte 2", "Descarte 2")
    assert stack.active is recorte2


# ---------- recorte aplicado a todas las capas visibles (feature 0.a) ---------- #

def _capa(nombre, pts, visible=True):
    from app.core.layers import CloudLayer
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(pts, float)))
    return CloudLayer(name=nombre, pcd=p, visible=visible)


def _stack_de_prueba():
    from app.core.layers import LayerStack
    s = LayerStack()
    s.layers = [
        _capa("A entera dentro", [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]),
        _capa("B partida",       [[0.3, 0.3, 0.3], [9.0, 9.0, 9.0]]),
        _capa("C entera fuera",  [[8.0, 8.0, 8.0], [9.0, 9.0, 9.0]]),
        _capa("D oculta",        [[0.4, 0.4, 0.4], [9.0, 9.0, 9.0]], visible=False),
    ]
    s.active_index = 0
    return s


def _dentro_de_la_caja(pcd):
    pts = np.asarray(pcd.points)
    return np.all((pts >= 0.0) & (pts <= 1.0), axis=1)


def test_split_visible_divide_solo_las_capas_partidas(tmp_path):
    """Las capas enteras de un lado no se dividen: con 35 scans, dividirlas
    igual llenaría la lista de capas vacías."""
    s = _stack_de_prueba()
    n_div, n_ocul, n_intacta = s.split_visible_fino(
        _dentro_de_la_caja, lambda f: None, tmp_path)
    assert (n_div, n_ocul, n_intacta) == (1, 1, 1)

    nombres = [c.name for c in s.layers]
    assert "A entera dentro" in nombres          # intacta
    assert "C entera fuera" in nombres           # sigue, pero oculta
    assert any(n.startswith("B partida · dentro") for n in nombres)
    assert any(n.startswith("B partida · fuera") for n in nombres)
    assert "B partida" not in nombres            # la fuente se consume


def test_split_visible_conserva_la_procedencia_por_scan(tmp_path):
    """Sin esto no se puede separar interior de exterior: cada mitad tiene que
    seguir diciendo de qué scan viene."""
    s = _stack_de_prueba()
    s.split_visible_fino(_dentro_de_la_caja, lambda f: None, tmp_path)
    hijas = [c.name for c in s.layers if "·" in c.name]
    assert all(h.startswith("B partida") for h in hijas)


def test_split_visible_no_toca_las_capas_ocultas(tmp_path):
    s = _stack_de_prueba()
    s.split_visible_fino(_dentro_de_la_caja, lambda f: None, tmp_path)
    oculta = [c for c in s.layers if c.name == "D oculta"]
    assert len(oculta) == 1 and len(oculta[0].pcd.points) == 2


def test_split_visible_deja_activa_una_capa_visible(tmp_path):
    s = _stack_de_prueba()
    s.split_visible_fino(_dentro_de_la_caja, lambda f: None, tmp_path)
    assert s.active.visible


def test_split_visible_falla_sin_modificar_si_no_conserva_nada(tmp_path):
    """Caja fuera de la nube: debe abortar limpio, no dejar el stack a medias."""
    s = _stack_de_prueba()
    antes = [c.name for c in s.layers]
    with pytest.raises(ValueError):
        s.split_visible_fino(lambda p: np.zeros(len(p.points), bool),
                             lambda f: None, tmp_path)
    assert [c.name for c in s.layers] == antes


def test_split_visible_recorta_tambien_la_nube_fina(tmp_path):
    """Si la capa tiene fine_path, las hijas deben quedar con su propio .ply."""
    s = _stack_de_prueba()
    fino = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
        np.array([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6], [9.0, 9.0, 9.0]])))
    ruta = tmp_path / "b_fino.ply"
    o3d.io.write_point_cloud(str(ruta), fino)
    s.layers[1].fine_path = ruta

    s.split_visible_fino(_dentro_de_la_caja, _dentro_de_la_caja, tmp_path / "edits")
    dentro = next(c for c in s.layers if c.name.endswith("dentro 1"))
    pts = np.asarray(o3d.io.read_point_cloud(str(dentro.fine_path)).points)
    assert len(pts) == 2 and np.all(pts <= 1.0)


# ---------- eliminación en bloque de los descartes ---------- #

def test_split_visible_marca_los_descartes(tmp_path):
    s = _stack_de_prueba()
    s.split_visible_fino(_dentro_de_la_caja, lambda f: None, tmp_path)
    # la mitad "fuera" y la capa que quedó entera fuera
    assert s.contar_descartes() == 2
    assert all(c.descarte is False for c in s.layers if c.visible)


def test_eliminar_descartes_borra_todas_de_una_vez(tmp_path):
    s = _stack_de_prueba()
    s.split_visible_fino(_dentro_de_la_caja, lambda f: None, tmp_path)
    n = s.eliminar_descartes()
    assert n == 2
    assert s.contar_descartes() == 0
    assert not any("· fuera" in c.name for c in s.layers)
    assert "D oculta" in [c.name for c in s.layers]   # oculta ≠ descarte


def test_eliminar_descartes_deja_una_activa_valida(tmp_path):
    s = _stack_de_prueba()
    s.split_visible_fino(_dentro_de_la_caja, lambda f: None, tmp_path)
    s.active_index = next(i for i, c in enumerate(s.layers) if c.descarte)
    s.eliminar_descartes()
    assert 0 <= s.active_index < len(s.layers)
    assert not s.active.descarte


def test_eliminar_descartes_no_vacia_el_proyecto():
    from app.core.layers import LayerStack
    s = LayerStack()
    s.layers = [_capa("solo descarte", [[0.0, 0.0, 0.0]])]
    s.layers[0].descarte = True
    s.active_index = 0
    with pytest.raises(ValueError):
        s.eliminar_descartes()
    assert len(s.layers) == 1
