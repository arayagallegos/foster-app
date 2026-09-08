"""tests/test_layer_dock.py — Dock de capas, separado del de herramientas.

La tabla tiene cinco columnas: Ver (visibilidad), Usar (capas activas), Capa,
En cache (puntos de trabajo) y En el archivo (puntos del escaneo original).
El resaltado azul NO decide nada por si mismo: solo elige sobre que filas actuan
los botones del panel.
"""
import sys

import numpy as np
import open3d as o3d
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from app.core.layers import CloudLayer, LayerStack
from app.gui.panels.layer_dock import COL_NOMBRE, COL_USAR, COL_VER


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture()
def dock(app):
    from app.gui.panels.layer_dock import LayerDock
    return LayerDock()


def _stack_con_split(n=100):
    """Stack tras un recorte: dos mitades que heredan el nombre de origen."""
    rng = np.random.default_rng(0)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rng.uniform(0, 1, (n, 3))))
    stack = LayerStack()
    stack.reset(pcd)
    keep = np.zeros(n, dtype=bool)
    keep[: n // 2] = True
    stack.split_activas(keep)
    return stack


def _stack_tres(n=30):
    s = LayerStack()
    for k in range(3):
        pts = np.zeros((n, 3))
        pts[:, 0] = np.arange(n) + 100 * k
        s.layers.append(CloudLayer(
            name=f"Scan {k}",
            pcd=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))))
    s.set_activas([0])
    return s


def _marcadas(dock, columna):
    return [i for i in range(dock._n_filas())
            if dock._item(i).checkState(columna) == Qt.CheckState.Checked]


def _seleccionar(dock, filas):
    dock._arbol.clearSelection()
    for f in filas:
        dock._item(f).setSelected(True)


# ------------------------------------------------------------------- basicos

def test_boton_restaurar_capa(dock):
    recibido = []
    dock.layer_restore_requested.connect(lambda: recibido.append(True))
    assert not dock._btn_restaurar.isEnabled()
    dock.set_restore_enabled(True)
    dock._btn_restaurar.click()
    assert recibido == [True]


def test_export_emite_senal(dock):
    recibido = []
    dock.export_requested.connect(lambda: recibido.append(True))
    dock._btn_export.click()
    assert recibido == [True]


def test_refresh_layers_muestra_capas_y_estado(dock):
    stack = _stack_con_split()
    dock.refresh_layers(stack)

    assert dock._n_filas() == 2
    assert "dentro 1" in dock._item(0).text(COL_NOMBRE)
    assert "fuera 1" in dock._item(1).text(COL_NOMBRE)
    assert "[descarte]" in dock._item(1).text(COL_NOMBRE)
    assert _marcadas(dock, COL_VER) == [0]        # la mitad descartada, oculta
    assert _marcadas(dock, COL_USAR) == [0]       # y no activa
    assert dock._btn_eliminar.isEnabled()


def test_refresh_una_capa_deshabilita_eliminar(dock):
    stack = LayerStack()
    stack.reset(o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.zeros((10, 3)))))
    dock.refresh_layers(stack)
    assert not dock._btn_eliminar.isEnabled()


def test_las_dos_columnas_son_independientes(dock):
    """Una capa puede verse sin usarse, y usarse sin verse."""
    stack = _stack_tres()
    stack.layers[1].visible = True
    stack.layers[2].visible = False
    stack.set_activas([2])
    dock.refresh_layers(stack)
    assert _marcadas(dock, COL_VER) == [0, 1]
    assert _marcadas(dock, COL_USAR) == [2]


# ------------------------------------------------------------------- columna Ver

def test_la_casilla_ver_emite_visibilidad(dock):
    stack = _stack_tres()
    dock.refresh_layers(stack)
    vis = []
    dock.layer_visibility_changed.connect(lambda i, v: vis.append((i, v)))
    _seleccionar(dock, [])
    dock._item(2).setCheckState(COL_VER, Qt.CheckState.Unchecked)
    assert vis == [(2, False)]


def test_la_casilla_ver_se_propaga_a_la_seleccion(dock):
    stack = _stack_tres()
    dock.refresh_layers(stack)
    vis = []
    dock.layers_visibility_changed.connect(lambda f, v: vis.append((list(f), v)))
    _seleccionar(dock, [0, 2])
    dock._item(0).setCheckState(COL_VER, Qt.CheckState.Unchecked)
    assert vis == [([0, 2], False)]
    assert _marcadas(dock, COL_VER) == [1]


# ------------------------------------------------------------------ columna Usar

def test_la_casilla_usar_define_las_capas_activas(dock):
    stack = _stack_tres()
    dock.refresh_layers(stack)
    act = []
    dock.layers_activated.connect(act.append)
    _seleccionar(dock, [])
    dock._item(2).setCheckState(COL_USAR, Qt.CheckState.Checked)
    assert act[-1] == [0, 2]          # el conjunto completo, no solo la tocada
    dock._item(0).setCheckState(COL_USAR, Qt.CheckState.Unchecked)
    assert act[-1] == [2]


def test_la_casilla_usar_se_propaga_a_la_seleccion(dock):
    stack = _stack_tres()
    dock.refresh_layers(stack)
    act = []
    dock.layers_activated.connect(act.append)
    _seleccionar(dock, [1, 2])
    dock._item(1).setCheckState(COL_USAR, Qt.CheckState.Checked)
    assert _marcadas(dock, COL_USAR) == [0, 1, 2]
    assert act[-1] == [0, 1, 2]


def test_seleccionar_filas_ya_no_cambia_las_activas(dock):
    """El cambio de fondo: el azul es transitorio y no debe decidir nada."""
    stack = _stack_tres()
    dock.refresh_layers(stack)
    act = []
    dock.layers_activated.connect(act.append)
    _seleccionar(dock, [1, 2])
    assert act == []
    assert _marcadas(dock, COL_USAR) == [0]


def test_marcar_activas_no_emite_ni_toca_la_seleccion(dock):
    stack = _stack_tres()
    dock.refresh_layers(stack)
    act = []
    dock.layers_activated.connect(act.append)
    _seleccionar(dock, [1, 2])
    dock.marcar_activas([0, 2])
    assert _marcadas(dock, COL_USAR) == [0, 2]
    assert dock._filas_seleccionadas() == [1, 2]
    assert act == []


def test_boton_usar_las_visibles(dock):
    recibido = []
    dock.activate_visible_requested.connect(lambda: recibido.append(True))
    dock._botones["/Usar lo que se ve"].click()
    assert recibido == [True]



# --------------------------------------------------------------------- borrado

def test_eliminar_se_lleva_toda_la_seleccion(dock):
    """Regresion: el boton usaba la fila actual y borraba una sola de doce."""
    stack = _stack_tres()
    dock.refresh_layers(stack)
    rem = []
    dock.layers_removed.connect(rem.append)
    _seleccionar(dock, [0, 2])
    dock._btn_eliminar.click()
    assert rem == [[0, 2]]


def test_el_boton_eliminar_dice_cuantas_capas_se_lleva(dock):
    stack = _stack_tres()
    dock.refresh_layers(stack)
    _seleccionar(dock, [0])
    assert dock._btn_eliminar.text() == "Eliminar"
    dock._item(1).setSelected(True)
    assert dock._btn_eliminar.text() == "Eliminar 2 capas"


def test_no_se_pueden_eliminar_todas_las_capas(dock):
    stack = _stack_tres()
    dock.refresh_layers(stack)
    dock._arbol.selectAll()
    assert not dock._btn_eliminar.isEnabled()


# ------------------------------------------------------------------- renombrar

def test_renombrar_emite_el_nombre_limpio(dock):
    """El texto visible lleva contador y marcas; el nombre real va aparte."""
    stack = _stack_tres()
    dock.refresh_layers(stack)
    nombres = []
    dock.layer_renamed.connect(lambda i, n: nombres.append((i, n)))
    dock._item(1).setData(COL_NOMBRE, Qt.ItemDataRole.UserRole, "Cúpula")
    assert nombres == [(1, "Cúpula")]


# ------------------------------------------------------------ clics de verdad

def test_clic_real_en_las_casillas(dock, qtbot):
    """Con clics reales sobre cada columna, cada casilla hace lo suyo."""
    from PyQt6.QtTest import QTest

    stack = _stack_tres()
    dock.refresh_layers(stack)
    qtbot.addWidget(dock)
    dock.show()
    arbol = dock._arbol

    def clic_casilla(fila, columna):
        rect = arbol.visualRect(arbol.indexFromItem(dock._item(fila), columna))
        QTest.mouseClick(arbol.viewport(), Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, rect.center())

    clic_casilla(1, COL_USAR)
    assert _marcadas(dock, COL_USAR) == [0, 1]
    clic_casilla(0, COL_VER)
    assert _marcadas(dock, COL_VER) == [1, 2]
    # Marcar 'Usar' no toco 'Ver' ni al reves.
    assert _marcadas(dock, COL_USAR) == [0, 1]


# ----------------------------------------------------- columnas de puntos

def test_las_cifras_van_en_columnas_no_pegadas_al_nombre(dock):
    """Regresion: el nombre traia '(170.634 pts) (29.245 pts)' sin etiquetas."""
    from app.gui.panels.layer_dock import COL_ARCHIVO, COL_PTS

    stack = _stack_tres(n=30)
    stack.layers[0].n_pts_archivo = 170_634
    dock.refresh_layers(stack)
    assert dock._item(0).text(COL_NOMBRE) == "Scan 0"       # el nombre, limpio
    assert dock._item(0).text(COL_PTS) == "30"
    assert dock._item(0).text(COL_ARCHIVO) == "170,634"
    assert dock._item(1).text(COL_ARCHIVO) == "-"           # capa sin origen


# ------------------------------------------------- atajos de la columna Usar

def _destinos(dock, ver, usar):
    dock._destinos["Ver"].setChecked(ver)
    dock._destinos["Usar"].setChecked(usar)
    dock._on_destino_cambiado()


def test_los_atajos_se_aplican_a_los_destinos_marcados(dock):
    """El caso que motivo el diseno: aislar y poner a trabajar en un clic."""
    stack = _stack_tres()
    dock.refresh_layers(stack)
    act, aislar = [], []
    dock.layers_activated.connect(act.append)
    dock.layers_isolate_requested.connect(aislar.append)

    _destinos(dock, ver=True, usar=True)
    _seleccionar(dock, [1, 2])
    dock._botones["/Solo esta"].click()
    assert aislar == [[1, 2]] and act[-1] == [1, 2]


def test_un_destino_solo_no_toca_al_otro(dock):
    stack = _stack_tres()
    dock.refresh_layers(stack)
    act, aislar = [], []
    dock.layers_activated.connect(act.append)
    dock.layers_isolate_requested.connect(aislar.append)

    _destinos(dock, ver=True, usar=False)
    _seleccionar(dock, [1])
    dock._botones["/Solo esta"].click()
    assert aislar == [[1]] and act == []
    assert _marcadas(dock, COL_USAR) == [0], "Usar no debia moverse"

    _destinos(dock, ver=False, usar=True)
    dock._botones["/Todas"].click()
    assert act[-1] == [0, 1, 2]
    assert len(aislar) == 1, "Ver no debia moverse"


def test_invertir_sobre_usar(dock):
    stack = _stack_tres()
    dock.refresh_layers(stack)
    _destinos(dock, ver=False, usar=True)
    act = []
    dock.layers_activated.connect(act.append)
    dock._botones["/Invertir"].click()
    assert _marcadas(dock, COL_USAR) == [1, 2] and act[-1] == [1, 2]


def test_invertir_usar_no_deja_el_conjunto_vacio(dock):
    """Sin capas en uso no hay herramienta posible."""
    stack = _stack_tres()
    dock.refresh_layers(stack)
    _destinos(dock, ver=False, usar=True)
    dock._botones["/Todas"].click()
    act = []
    dock.layers_activated.connect(act.append)
    dock._botones["/Invertir"].click()          # el complemento seria vacio
    assert _marcadas(dock, COL_USAR) == [0, 1, 2]
    assert act == []


def test_ninguna_vacia_el_conjunto_en_uso(dock):
    """Vaciar es un paso legitimo: se limpia para rearmar otro conjunto."""
    dock.refresh_layers(_stack_tres())
    _destinos(dock, ver=False, usar=True)
    dock._botones["/Todas"].click()
    act = []
    dock.layers_activated.connect(act.append)
    dock._botones["/Ninguna"].click()
    assert act == [[]]
    assert _marcadas(dock, COL_USAR) == []


def test_no_se_pueden_apagar_los_dos_destinos(dock):
    """Sin destino los atajos no harian nada: el ultimo se vuelve a encender."""
    dock.refresh_layers(_stack_tres())
    _destinos(dock, ver=True, usar=True)
    dock._destinos["Ver"].setChecked(False)
    dock._destinos["Ver"].clicked.emit(False)
    dock._destinos["Usar"].setChecked(False)
    dock._destinos["Usar"].click()
    assert dock._destinos["Ver"].isChecked() or dock._destinos["Usar"].isChecked()


def test_el_segmento_de_destino_no_es_excluyente(dock):
    """Van pegados y parecen un interruptor, pero admiten los dos encendidos."""
    ver, usar = dock._destinos["Ver"], dock._destinos["Usar"]
    assert ver.isCheckable() and usar.isCheckable()
    assert ver.isChecked() and usar.isChecked(), "por defecto, los dos"

    # Ni Qt ni un grupo de botones deben apagar uno al encender el otro.
    assert ver.group() is None and usar.group() is None
    assert not ver.autoExclusive() and not usar.autoExclusive()

    _destinos(dock, ver=False, usar=True)
    assert not ver.isChecked() and usar.isChecked()
    ver.setChecked(True)
    assert ver.isChecked() and usar.isChecked(), "encender uno no apaga el otro"


def test_el_segmento_va_pegado(dock):
    """Sin hueco entre ambos: es lo que los hace leer como un solo control."""
    contenedor = dock._destinos["Ver"].parentWidget()
    assert dock._destinos["Usar"].parentWidget() is contenedor
    assert contenedor.layout().spacing() == 0
