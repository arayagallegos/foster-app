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


def test_refresh_clusters_puebla_la_lista(dock):
    """Poblar la lista de DBSCAN no depende de que se ejecute DBSCAN.

    La primera versión importaba `QListWidget` pero no `QListWidgetItem`, y como
    el único camino que construye items es este, el fallo solo aparecía al
    terminar un agrupamiento real sobre una nube cargada.
    """
    from app.modules.clusters import InfoCluster

    infos = [
        InfoCluster(id=0, n_pts=1200, planaridad=0.91,
                    extension=np.array([3.0, 2.0, 0.4]),
                    centro=np.zeros(3)),
        InfoCluster(id=1, n_pts=80, planaridad=0.12,
                    extension=np.array([0.5, 0.5, 4.0]),
                    centro=np.ones(3)),
    ]
    dock.refresh_clusters(infos)

    assert dock._lista_clusters.count() == 2
    assert dock._lista_clusters.item(0).data(Qt.ItemDataRole.UserRole) == 0
    assert "1,200 pts" in dock._lista_clusters.item(0).text()
    # Sin selección no hay nada que capturar ni eliminar
    assert not dock._btn_db_capturar.isEnabled()
    assert not dock._btn_db_eliminar.isEnabled()

    dock.refresh_clusters([])
    assert dock._lista_clusters.count() == 0
