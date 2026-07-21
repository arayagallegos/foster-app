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
