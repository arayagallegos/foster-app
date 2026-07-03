import pytest
import sys
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(sys.argv)


def test_crop_panel_has_lasso_signals(app):
    from app.gui.panels.crop_panel import CropPanel
    panel = CropPanel("test.ply")
    assert hasattr(panel, "lasso_started")
    assert hasattr(panel, "lasso_apply_requested")
    assert hasattr(panel, "lasso_cancel_requested")


def test_lasso_apply_and_cancel_start_disabled(app):
    from app.gui.panels.crop_panel import CropPanel
    panel = CropPanel("test.ply")
    assert not panel._btn_apply_lasso.isEnabled()
    assert not panel._btn_cancel_lasso.isEnabled()


def test_set_lasso_active_enables_cancel_disables_start(app):
    from app.gui.panels.crop_panel import CropPanel
    panel = CropPanel("test.ply")
    panel.set_lasso_active(True)
    assert not panel._btn_start_lasso.isEnabled()
    assert panel._btn_cancel_lasso.isEnabled()


def test_set_lasso_active_false_restores_start(app):
    from app.gui.panels.crop_panel import CropPanel
    panel = CropPanel("test.ply")
    panel.set_lasso_active(True)
    panel.set_lasso_active(False)
    assert panel._btn_start_lasso.isEnabled()
    assert not panel._btn_cancel_lasso.isEnabled()


def test_set_lasso_has_selection_enables_apply(app):
    from app.gui.panels.crop_panel import CropPanel
    panel = CropPanel("test.ply")
    assert not panel._btn_apply_lasso.isEnabled()
    panel.set_lasso_has_selection(True)
    assert panel._btn_apply_lasso.isEnabled()
    panel.set_lasso_has_selection(False)
    assert not panel._btn_apply_lasso.isEnabled()


def test_lasso_started_emits_selected_set_op(app, qtbot):
    from app.gui.panels.crop_panel import CropPanel
    panel = CropPanel("test.ply")
    received = []
    panel.lasso_started.connect(received.append)
    panel._btn_start_lasso.click()
    assert received == ["union"]
