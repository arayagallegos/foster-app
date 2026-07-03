"""tests/test_viewer_lasso.py — Reproducción del cierre del lazo en el viewer real."""
import sys

import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture()
def viewer(app, qtbot):
    from app.gui.viewer import Viewer3D

    v = Viewer3D()
    qtbot.addWidget(v)
    v.resize(800, 600)
    v.show()
    qtbot.waitExposed(v)
    yield v
    v.plotter.close()


def _add_vertices(qtbot, widget, posiciones):
    for x, y in posiciones:
        qtbot.mouseClick(widget, Qt.MouseButton.LeftButton, pos=QPoint(x, y))


def test_lasso_clic_izquierdo_agrega_vertices(viewer, qtbot):
    viewer.start_lasso(lambda verts: None)
    _add_vertices(qtbot, viewer.plotter, [(100, 100), (300, 100), (200, 300)])
    assert len(viewer._lasso_screen_verts) == 3


def test_lasso_clic_derecho_cierra_y_llama_callback(viewer, qtbot):
    recibido = []
    viewer.start_lasso(recibido.append)
    _add_vertices(qtbot, viewer.plotter, [(100, 100), (300, 100), (200, 300)])

    qtbot.mouseClick(viewer.plotter, Qt.MouseButton.RightButton, pos=QPoint(200, 150))

    assert len(recibido) == 1, "el clic derecho no cerró el lazo"
    assert len(recibido[0]) == 3


def test_lasso_clic_sobre_primer_vertice_cierra(viewer, qtbot):
    """Unir el inicio con el final: clicar cerca del primer vértice cierra el lazo."""
    recibido = []
    viewer.start_lasso(recibido.append)
    _add_vertices(qtbot, viewer.plotter, [(100, 100), (300, 100), (200, 300)])

    qtbot.mouseClick(viewer.plotter, Qt.MouseButton.LeftButton, pos=QPoint(103, 102))

    assert len(recibido) == 1, "clicar el primer vértice no cerró el lazo"
    assert len(recibido[0]) == 3


def test_camara_se_libera_al_cerrar_el_lazo(viewer, qtbot):
    """Tras cerrar el polígono debe poder moverse la cámara (estilo restaurado)."""
    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser

    viewer.start_lasso(lambda verts: None)
    iren = viewer.plotter.iren.interactor
    assert isinstance(iren.GetInteractorStyle(), vtkInteractorStyleUser)

    _add_vertices(qtbot, viewer.plotter, [(100, 100), (300, 100), (200, 300)])
    qtbot.mouseClick(viewer.plotter, Qt.MouseButton.RightButton, pos=QPoint(200, 150))

    assert not isinstance(iren.GetInteractorStyle(), vtkInteractorStyleUser), (
        "la cámara sigue bloqueada tras cerrar el lazo"
    )
    assert viewer._lasso_obs_ids == []


def test_lasso_doble_clic_cierra(viewer, qtbot):
    """La barra de estado promete doble clic para cerrar: debe cumplirse."""
    recibido = []
    viewer.start_lasso(recibido.append)
    _add_vertices(qtbot, viewer.plotter, [(100, 100), (300, 100), (200, 300)])

    qtbot.mouseDClick(viewer.plotter, Qt.MouseButton.LeftButton, pos=QPoint(200, 150))

    assert len(recibido) == 1, "el doble clic no cerró el lazo"
    assert len(recibido[0]) >= 3
