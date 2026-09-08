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


def test_preview_split_en_viewer_aislado(app, qtbot):
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
    v.preview_split(pcd, mask, "cloud_lidar")

    assert "_preview_keep" in v.plotter.actors, "falta el actor verde (conserva)"
    assert "_preview_discard" in v.plotter.actors, "falta el actor rojo (elimina)"
    nube = np.array(v.plotter.actors["cloud_lidar"].bounds)
    for name in ("_preview_keep", "_preview_discard"):
        b = np.array(v.plotter.actors[name].bounds)
        assert b[0] >= nube[0] - 1 and b[1] <= nube[1] + 1, (
            f"{name} fuera de la nube: {b} vs {nube}"
        )
    assert v.plotter.actors["cloud_lidar"].GetProperty().GetOpacity() < 0.5

    v.clear_preview()
    assert "_preview_keep" not in v.plotter.actors
    assert "_preview_discard" not in v.plotter.actors
    assert v.plotter.actors["cloud_lidar"].GetProperty().GetOpacity() == 1.0
    v.plotter.close()


def _mainwindow_con_nube(qtbot):
    from app.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1200, 800)
    w.show()
    qtbot.waitExposed(w)

    pcd = _nube_utm()
    w.project.set_lidar(pcd, "test.ply")
    w.viewer.show_cloud(pcd, name="cloud_lidar", point_size=4.0)
    return w


def _dibujar_lazo(qtbot, plotter, vertices, cierre=(500, 300)):
    for x, y in vertices:
        qtbot.mouseClick(plotter, Qt.MouseButton.LeftButton, pos=QPoint(x, y))
    qtbot.mouseClick(plotter, Qt.MouseButton.RightButton, pos=QPoint(*cierre))


def test_flujo_lazo_preview_y_aplicar_crea_capas(app, qtbot):
    w = _mainwindow_con_nube(qtbot)
    plotter = w.viewer.plotter

    w._activate_tool("lazo")
    assert w._layer_stack is not None and len(w._layer_stack) == 1

    w._on_lasso_started("union")
    _dibujar_lazo(qtbot, plotter, [(200, 150), (900, 150), (900, 600), (200, 600)])

    assert w._lasso_mask is not None and w._lasso_mask.any(), "el lazo no seleccionó nada"
    assert "_preview_keep" in plotter.actors, "no se creó el preview verde"
    assert "_preview_discard" in plotter.actors, "no se creó el preview rojo"

    # Aplicar: la capa se reemplaza por sus dos mitades, que heredan su nombre.
    # No se guarda una tercera copia de la original porque la union de las dos
    # mitades ya es exactamente la original.
    w._on_lasso_apply()
    stack = w._layer_stack
    assert len(stack) == 2
    assert stack.active.name == "Original · dentro 1" and stack.active.visible
    assert stack.layers[1].name == "Original · fuera 1"
    assert not stack.layers[1].visible and stack.layers[1].descarte
    assert "layer_0" in plotter.actors               # la mitad capturada se dibuja
    assert "layer_1" not in plotter.actors           # ocultas no se dibujan
    assert "_preview_keep" not in plotter.actors    # preview limpiado
    plotter.close()


def test_quitar_sin_seleccion_previa_parte_de_toda_la_capa(app, qtbot):
    """'Quitar' con seleccion vacia debe significar 'todo menos el lazo'."""
    w = _mainwindow_con_nube(qtbot)
    plotter = w.viewer.plotter

    w._activate_tool("lazo")
    w._on_lasso_started("difference")
    # Lazo chico en una esquina: se debe quitar SOLO eso del total
    _dibujar_lazo(qtbot, plotter, [(300, 200), (500, 200), (500, 400), (300, 400)],
                  cierre=(400, 300))

    assert w._lasso_mask is not None
    n_sel, n_total = int(w._lasso_mask.sum()), len(w._lasso_mask)
    assert 0 < n_sel < n_total, "debe quedar seleccionado todo menos el lazo"
    assert n_sel > n_total * 0.5, "la mayoria de la capa debe seguir seleccionada"
    plotter.close()
