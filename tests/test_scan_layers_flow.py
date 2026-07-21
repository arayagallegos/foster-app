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
