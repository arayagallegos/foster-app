"""tests/test_crop_dock.py — Señales y estados del dock de recorte."""
import sys

import numpy as np
import open3d as o3d
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from app.core.layers import LayerStack


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture()
def dock(app):
    from app.gui.panels.crop_dock import CropDock
    return CropDock()


def _stack_con_split(n=100):
    """Stack tras un recorte: Original (oculta), Recorte 1 (activa), Descarte 1."""
    rng = np.random.default_rng(0)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rng.uniform(0, 1, (n, 3))))
    stack = LayerStack()
    stack.reset(pcd)
    keep = np.zeros(n, dtype=bool)
    keep[: n // 2] = True
    stack.split_active(keep)
    return stack


def test_set_tool_cambia_pagina(dock):
    dock.set_tool("caja")
    assert dock._stack.currentIndex() == 0
    dock.set_tool("lazo")
    assert dock._stack.currentIndex() == 1
    with pytest.raises(ValueError):
        dock.set_tool("tijera")


def test_estados_iniciales_lazo(dock):
    assert not dock._btn_apply_lasso.isEnabled()
    assert not dock._btn_cancel_lasso.isEnabled()
    dock.set_lasso_active(True)
    assert not dock._btn_start_lasso.isEnabled()
    assert dock._btn_cancel_lasso.isEnabled()
    dock.set_lasso_has_selection(np.bool_(True))  # numpy.bool_ aceptado
    assert dock._btn_apply_lasso.isEnabled()


def test_lasso_started_emite_operacion(dock):
    recibido = []
    dock.lasso_started.connect(recibido.append)
    dock._lasso_op_group.buttons()[1].setChecked(True)  # Quitar lo del lazo
    dock._btn_start_lasso.click()
    assert recibido == ["difference"]
    # La explicación vive en el tooltip del radio
    assert "ROJO" in dock._lasso_op_group.buttons()[1].toolTip()
