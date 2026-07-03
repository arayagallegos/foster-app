"""tests/test_lasso_highlight.py — Reproducción: el resaltado amarillo del lazo."""
import sys

import numpy as np
import open3d as o3d
import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(sys.argv)


def _nube_utm(n=2000, seed=0):
    """Nube con coordenadas grandes tipo UTM, como la nube real del Observatorio."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 10, (n, 3)) + np.array([350_000.0, 6_300_000.0, 700.0])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    return pcd


def test_highlight_selection_en_viewer_aislado(app, qtbot):
    from app.gui.viewer import Viewer3D

    v = Viewer3D()
    qtbot.addWidget(v)
    v.resize(800, 600)
    v.show()
    qtbot.waitExposed(v)

    pcd = _nube_utm()
    v.show_cloud(pcd, name="cloud_lidar")

    mask = np.zeros(len(pcd.points), dtype=bool)
    mask[:500] = True
    v.highlight_selection(pcd, mask, "cloud_lidar")

    assert "_lasso_selection" in v.plotter.actors, "no se creó el actor amarillo"
    sel = np.array(v.plotter.actors["_lasso_selection"].bounds)
    nube = np.array(v.plotter.actors["cloud_lidar"].bounds)
    # El amarillo debe quedar DENTRO del volumen de la nube mostrada
    assert sel[0] >= nube[0] - 1 and sel[1] <= nube[1] + 1, (
        f"amarillo fuera de la nube: sel={sel}, nube={nube}"
    )
    v.plotter.close()


def test_flujo_completo_mainwindow_muestra_amarillo(app, qtbot):
    from app.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1200, 800)
    w.show()
    qtbot.waitExposed(w)

    # Inyectar una nube como si se hubiera cargado
    pcd = _nube_utm()
    w.project.set_lidar(pcd, "test.ply")
    w.viewer.show_cloud(pcd, name="cloud_lidar", point_size=4.0)

    # El panel de recorte existe en el flujo real (bug: numpy.bool → setEnabled)
    from app.gui.panels.crop_panel import CropPanel
    w._crop_panel = CropPanel("test.ply", parent=w)

    # Iniciar lazo y dibujar un polígono grande en pantalla
    w._on_lasso_started("union")
    plotter = w.viewer.plotter
    for x, y in [(200, 150), (900, 150), (900, 600), (200, 600)]:
        qtbot.mouseClick(plotter, Qt.MouseButton.LeftButton, pos=QPoint(x, y))
    qtbot.mouseClick(plotter, Qt.MouseButton.RightButton, pos=QPoint(500, 300))

    assert w._lasso_mask is not None and w._lasso_mask.any(), "el lazo no seleccionó nada"
    assert "_lasso_selection" in plotter.actors, "no se creó el actor amarillo"

    sel = np.array(plotter.actors["_lasso_selection"].bounds)
    nube = np.array(plotter.actors["cloud_lidar"].bounds)
    assert sel[0] >= nube[0] - 1 and sel[1] <= nube[1] + 1, (
        f"amarillo fuera de la nube: sel={sel}, nube={nube}"
    )
    plotter.close()


def test_quitar_sin_seleccion_previa_parte_de_toda_la_nube(app, qtbot):
    """'Quitar' con seleccion vacia debe significar 'todo menos el lazo'."""
    from app.gui.main_window import MainWindow
    from app.gui.panels.crop_panel import CropPanel

    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1200, 800)
    w.show()
    qtbot.waitExposed(w)

    pcd = _nube_utm()
    w.project.set_lidar(pcd, "test.ply")
    w.viewer.show_cloud(pcd, name="cloud_lidar", point_size=4.0)
    w._crop_panel = CropPanel("test.ply", parent=w)

    w._on_lasso_started("difference")
    plotter = w.viewer.plotter
    # Lazo chico en una esquina: se debe quitar SOLO eso del total
    for x, y in [(300, 200), (500, 200), (500, 400), (300, 400)]:
        qtbot.mouseClick(plotter, Qt.MouseButton.LeftButton, pos=QPoint(x, y))
    qtbot.mouseClick(plotter, Qt.MouseButton.RightButton, pos=QPoint(400, 300))

    assert w._lasso_mask is not None
    n_sel, n_total = int(w._lasso_mask.sum()), len(w._lasso_mask)
    assert 0 < n_sel < n_total, "debe quedar seleccionado todo menos el lazo"
    assert n_sel > n_total * 0.5, "la mayoria de la nube debe seguir seleccionada"
    plotter.close()
