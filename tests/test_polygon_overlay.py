import pytest
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QPoint


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(sys.argv)


def test_polygon_overlay_starts_hidden(app):
    from app.gui.polygon_overlay import PolygonOverlay
    overlay = PolygonOverlay()
    assert not overlay.isVisible()


def test_polygon_overlay_activate_shows_widget(app):
    from app.gui.polygon_overlay import PolygonOverlay
    overlay = PolygonOverlay()
    overlay.activate()
    assert overlay.isVisible()
    overlay.deactivate()


def test_polygon_overlay_deactivate_hides_and_clears(app):
    from app.gui.polygon_overlay import PolygonOverlay
    overlay = PolygonOverlay()
    overlay.activate()
    overlay._vertices = [QPoint(0, 0), QPoint(10, 0), QPoint(5, 10)]
    overlay.deactivate()
    assert not overlay.isVisible()
    assert overlay._vertices == []


def test_polygon_closed_signal_emitted_with_3_or_more_verts(app, qtbot):
    from app.gui.polygon_overlay import PolygonOverlay
    overlay = PolygonOverlay()
    overlay.activate()
    overlay._vertices = [QPoint(0, 0), QPoint(100, 0), QPoint(50, 100)]

    received = []
    overlay.polygon_closed.connect(received.append)
    overlay._close_polygon()

    assert len(received) == 1
    assert received[0] == [(0, 0), (100, 0), (50, 100)]
    assert not overlay.isVisible()


def test_polygon_close_ignored_with_fewer_than_3_verts(app, qtbot):
    from app.gui.polygon_overlay import PolygonOverlay
    overlay = PolygonOverlay()
    overlay.activate()
    overlay._vertices = [QPoint(0, 0), QPoint(100, 0)]

    received = []
    overlay.polygon_closed.connect(received.append)
    overlay._close_polygon()

    assert len(received) == 0
    assert not overlay.isVisible()
