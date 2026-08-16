"""tests/test_scan_layers_flow.py — Flujo de carga de scans como capas en la GUI."""
import sys

import numpy as np
import open3d as o3d
import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(sys.argv)


def _scan_info(idx, n, tmp_path):
    from app.core.io import ScanCacheInfo
    rng = np.random.default_rng(idx)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rng.uniform(0, 5, (n, 3))))
    fp = tmp_path / f"scan_{idx:02d}.ply"
    o3d.io.write_point_cloud(str(fp), pcd)
    return ScanCacheInfo(index=idx, n_pts_fino=n, fine_path=fp, pcd_grueso=pcd)


def test_activar_recorte_funciona_con_stack_de_scans(app, qtbot, tmp_path):
    """Regresión: con capas-scan cargadas (project.lidar_cloud vacío), activar la
    herramienta de recorte NO debe abortar por no encontrar nube en el project."""
    from app.core.layers import CloudLayer, LayerStack
    from app.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    rng = np.random.default_rng(0)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rng.uniform(0, 5, (300, 3))))
    stack = LayerStack()
    stack.layers = [CloudLayer(name="Scan 00", pcd=pcd, fine_path=tmp_path / "s.ply")]
    stack.active_index = 0
    w._layer_stack = stack
    assert w.project.lidar_cloud is None       # el flujo de scans no lo llena

    assert w._ensure_layer_stack() is True     # antes devolvía False → recorte abortaba
    w._activate_tool("caja")
    assert w._active_tool == "caja"            # la herramienta quedó activa
    w.close()


def test_recorte_caja_preserva_fino_en_mainwindow(app, qtbot, tmp_path):
    from app.core.layers import CloudLayer, LayerStack
    from app.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    rng = np.random.default_rng(0)
    fino = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(rng.uniform(0, 10, (4000, 3))))
    fp = tmp_path / "scan_00.ply"
    o3d.io.write_point_cloud(str(fp), fino)
    grueso = fino.voxel_down_sample(0.5)
    stack = LayerStack()
    stack.layers = [CloudLayer(name="Scan 00", pcd=grueso, fine_path=fp)]
    stack.active_index = 0
    w._layer_stack = stack
    w._active_tool = "caja"
    w._crop_min = np.array([0., 0., 0.])
    w._crop_max = np.array([5., 10., 10.])
    w._EDITS_DIR = tmp_path / "_edits"

    w._on_box_apply()
    # la nueva capa activa (Recorte) conserva fine_path fino recortado
    assert w._layer_stack.active.fine_path is not None
    assert w._layer_stack.active.fine_path.exists()
    recorte_fino = o3d.io.read_point_cloud(str(w._layer_stack.active.fine_path))
    assert np.all(np.asarray(recorte_fino.points)[:, 0] <= 5.0)
    w.close()


def test_on_load_lidar_enruta_segun_scans(app, qtbot, monkeypatch):
    """.e57 multi-scan → capas; otros archivos → nube única. (Una sola MainWindow
    para no acumular contextos VTK en el proceso de test.)"""
    from app.gui import main_window as mw
    w = mw.MainWindow()
    qtbot.addWidget(w)
    llamadas = {"scans": 0, "unica": 0}
    monkeypatch.setattr(w, "_cargar_scans_por_capas",
                        lambda p: llamadas.__setitem__("scans", llamadas["scans"] + 1))
    monkeypatch.setattr(w, "_cargar_nube_unica",
                        lambda p: llamadas.__setitem__("unica", llamadas["unica"] + 1))

    # .e57 con 5 scans → capas
    monkeypatch.setattr(w, "_open_file_dialog", lambda *a, **k: "X.e57")
    monkeypatch.setattr(mw, "_e57_scan_count_safe", lambda p: 5)
    w._on_load_lidar()
    assert (llamadas["scans"], llamadas["unica"]) == (1, 0)

    # .ply → nube única
    monkeypatch.setattr(w, "_open_file_dialog", lambda *a, **k: "nube.ply")
    monkeypatch.setattr(mw, "_e57_scan_count_safe", lambda p: 1)
    w._on_load_lidar()
    assert (llamadas["scans"], llamadas["unica"]) == (1, 1)
    w.close()


def test_scan_layers_loaded_puebla_stack(app, qtbot, tmp_path):
    from app.gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    scans = [_scan_info(0, 500, tmp_path), _scan_info(1, 800, tmp_path)]
    w._on_scan_layers_loaded(scans)
    assert w._layer_stack is not None
    assert len(w._layer_stack) == 2
    assert w._layer_stack.layers[0].fine_path == scans[0].fine_path
    assert "Scan 00" in w._layer_stack.layers[0].name
    w.close()
