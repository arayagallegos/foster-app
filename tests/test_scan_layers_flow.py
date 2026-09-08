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


def test_marcar_usar_en_dos_capas_las_activa_y_el_corte_divide_ambas(app, qtbot):
    """Extremo a extremo: la columna 'Usar' es el conjunto activo."""
    from app.core.layers import CloudLayer, LayerStack
    from app.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    stack = LayerStack()
    for k in range(3):
        pts = np.zeros((20, 3))
        pts[:, 0] = np.arange(20) * 0.1 + k        # cada scan en su franja de x
        stack.layers.append(CloudLayer(
            name=f"Scan {k}",
            pcd=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))))
    stack.set_activas([0])
    w._layer_stack = stack
    dock = w._ensure_layer_dock()
    dock.refresh_layers(stack)

    # El usuario marca 'Usar' en la segunda capa.
    from PyQt6.QtCore import Qt

    from app.gui.panels.layer_dock import COL_USAR
    dock._item(1).setCheckState(COL_USAR, Qt.CheckState.Checked)
    assert stack.active_indices == [0, 1]
    assert len(stack.active.pcd.points) == 40      # la union de ambas

    # Un corte que conserva la mitad de cada capa activa.
    keep = np.asarray(stack.active.pcd.points)[:, 0] < 1.5
    w._apply_split(keep)

    nombres = [c.name for c in stack.layers]
    assert nombres == ["Scan 0 · dentro 1", "Scan 0 · fuera 1",
                       "Scan 1 · dentro 1", "Scan 1 · fuera 1", "Scan 2"]
    assert stack.active_indices == [0, 2]          # las dos mitades capturadas
    assert len(stack.layers[4].pcd.points) == 20   # la no activa quedo intacta


def test_eliminar_varias_capas_y_restaurarlas_todas(app, qtbot, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from app.core.layers import CloudLayer, LayerStack
    from app.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    stack = LayerStack()
    for k in range(4):
        stack.layers.append(CloudLayer(
            name=f"Scan {k}",
            pcd=o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(np.zeros((10 + k, 3))))))
    stack.set_activas([0])
    w._layer_stack = stack
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)

    w._on_layers_removed([0, 2])
    assert [c.name for c in stack.layers] == ["Scan 1", "Scan 3"]

    w._on_layer_restore()
    assert [c.name for c in stack.layers] == ["Scan 0", "Scan 1", "Scan 2", "Scan 3"]
    assert [len(c.pcd.points) for c in stack.layers] == [10, 11, 12, 13]


def test_no_se_eliminan_todas_las_capas(app, qtbot, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from app.core.layers import CloudLayer, LayerStack
    from app.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    stack = LayerStack()
    for k in range(2):
        stack.layers.append(CloudLayer(
            name=f"Scan {k}",
            pcd=o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(np.zeros((10, 3))))))
    stack.set_activas([0])
    w._layer_stack = stack
    pedido = []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: pedido.append(True))

    w._on_layers_removed([0, 1])
    assert len(stack.layers) == 2
    assert pedido == [], "no debe ni preguntar si la accion es imposible"


def test_usar_las_visibles_copia_una_columna_en_la_otra(app, qtbot):
    """El caso frecuente 'operar sobre lo que veo', en un clic."""
    from PyQt6.QtCore import Qt

    from app.core.layers import CloudLayer, LayerStack
    from app.gui.main_window import MainWindow
    from app.gui.panels.layer_dock import COL_USAR

    w = MainWindow()
    qtbot.addWidget(w)
    stack = LayerStack()
    for k in range(5):
        stack.layers.append(CloudLayer(
            name=f"Scan {k}", visible=k in (1, 3, 4),
            pcd=o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(np.zeros((10, 3))))))
    stack.set_activas([0])
    w._layer_stack = stack
    dock = w._ensure_layer_dock()
    dock.refresh_layers(stack)

    dock._botones["/Usar lo que se ve"].click()
    assert stack.active_indices == [1, 3, 4]
    assert [i for i in range(5)
            if dock._item(i).checkState(COL_USAR) == Qt.CheckState.Checked] == [1, 3, 4]


def test_usar_las_visibles_sin_capas_visibles_no_deja_el_stack_vacio(app, qtbot):
    from app.core.layers import CloudLayer, LayerStack
    from app.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    stack = LayerStack()
    for k in range(3):
        stack.layers.append(CloudLayer(
            name=f"Scan {k}", visible=False,
            pcd=o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(np.zeros((10, 3))))))
    stack.set_activas([1])
    w._layer_stack = stack
    w._ensure_layer_dock().refresh_layers(stack)

    w._on_activate_visible()
    assert stack.active_indices == [1], "sin visibles, la activa no debe perderse"


def test_el_resumen_en_pantalla_sigue_a_la_visibilidad(app, qtbot):
    """Regresion: se fijaba al cargar y seguia anunciando los puntos iniciales."""
    from app.core.layers import CloudLayer, LayerStack
    from app.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    stack = LayerStack()
    for k in range(4):
        stack.layers.append(CloudLayer(
            name=f"Scan {k}",
            pcd=o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(np.zeros((100, 3))))))
    stack.set_activas([0])
    w._layer_stack = stack
    w._ensure_layer_dock().refresh_layers(stack)

    w._redibujar_capas(stack)
    texto = w.info_panel._etiqueta_vista.text()
    assert "4 de 4 capas" in texto and "400 de 400 pts" in texto

    w._on_layer_visibility(0, False)
    w._on_layer_visibility(1, False)
    texto = w.info_panel._etiqueta_vista.text()
    assert "2 de 4 capas" in texto and "200 de 400 pts" in texto

    # Y el conjunto activo tambien se refleja, aparte de la visibilidad.
    w._on_layers_activated([0, 2, 3])
    texto = w.info_panel._etiqueta_vista.text()
    assert "3 capas" in texto and "300 pts" in texto


def test_un_panel_cerrado_se_recupera_desde_el_menu(app, qtbot):
    """Regresion: los docks eran cerrables y no habia forma de reabrirlos."""
    from PyQt6.QtWidgets import QMenu

    from app.core.layers import CloudLayer, LayerStack
    from app.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    stack = LayerStack()
    stack.layers = [CloudLayer(
        name="Scan 0",
        pcd=o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(np.zeros((10, 3)))))]
    stack.set_activas([0])
    w._layer_stack = stack
    dock = w._ensure_layer_dock()
    dock.refresh_layers(stack)
    dock.show()
    assert not dock.isHidden()

    dock.close()                       # el usuario pulsa la X
    assert dock.isHidden()

    menu = QMenu()
    w._poblar_menu_paneles(menu)
    accion = next(a for a in menu.actions() if a.text() == "Capas")
    accion.trigger()
    assert not dock.isHidden(), "el panel debe volver, no quedarse perdido"
    # Y conserva su contenido: no se reconstruye desde cero.
    assert dock._n_filas() == 1


def test_el_menu_de_paneles_no_crea_el_dock_de_herramienta(app, qtbot):
    """Crearlo seria mostrarlo, y debe aparecer solo al usar una herramienta."""
    from PyQt6.QtWidgets import QMenu

    from app.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    menu = QMenu()
    w._poblar_menu_paneles(menu)
    assert getattr(w, "_crop_dock", None) is None
    # Informacion existe siempre; Herramienta no aparece hasta usarla.
    textos = [a.text() for a in menu.actions() if a.text()]
    assert "Información" in textos
    assert "Herramienta" not in textos


def _ventana_con_capas(qtbot, n=3):
    from app.core.layers import CloudLayer, LayerStack
    from app.gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    stack = LayerStack()
    for k in range(n):
        stack.layers.append(CloudLayer(
            name=f"Scan {k}",
            pcd=o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(np.zeros((10, 3))))))
    stack.set_activas([0])
    w._layer_stack = stack
    w._ensure_layer_dock().refresh_layers(stack)
    return w, stack


def test_vaciar_el_conjunto_en_uso_cierra_la_herramienta(app, qtbot):
    """Vaciar es legitimo, pero deja la herramienta sin material."""
    w, stack = _ventana_con_capas(qtbot)
    w._activate_tool("lazo")
    assert w._active_tool == "lazo"

    w._on_layers_activated([])
    assert stack.active_indices == []
    assert w._active_tool is None, "la herramienta no debe quedar apuntando a nada"


def test_sin_capas_en_uso_no_se_puede_activar_una_herramienta(app, qtbot):
    w, stack = _ventana_con_capas(qtbot)
    stack.set_activas([])
    w._activate_tool("lazo")
    assert w._active_tool is None
    assert "Usar" in w._status_label.text()

    # Y al volver a marcar una capa, la herramienta vuelve a estar disponible.
    w._on_layers_activated([1])
    w._activate_tool("lazo")
    assert w._active_tool == "lazo"


def test_el_conjunto_vacio_no_se_traduce_en_la_ultima_capa(app, qtbot):
    """Regresion: `stack.active` devolvia layers[-1] en silencio."""
    import pytest as _pytest

    w, stack = _ventana_con_capas(qtbot)
    stack.set_activas([])
    with _pytest.raises(ValueError, match="ninguna capa en uso"):
        _ = stack.active


def test_cambiar_visibilidad_sin_capas_en_uso_no_revienta(app, qtbot):
    """Regresion reportada: IndexError en activas[0] al tocar la visibilidad
    despues de vaciar el conjunto en uso."""
    w, stack = _ventana_con_capas(qtbot, n=4)
    w._on_layers_activated([])
    assert stack.active_indices == []

    w._on_layers_visibility([0, 1], False)      # el camino del traceback
    w._on_layer_visibility(2, False)
    assert "Ninguna capa en uso" in w._status_label.text()

    # Y con capas en uso el mensaje sigue siendo el util.
    w._on_layers_activated([1])
    w._on_layer_visibility(2, True)
    assert "Scan 1" in w._status_label.text()
    w._on_layers_activated([0, 1])
    w._on_layer_visibility(3, True)
    assert "2 capas en uso" in w._status_label.text()


def test_vaciar_el_conjunto_oculta_el_panel_de_herramienta(app, qtbot):
    """Sus controles seguirian emitiendo contra un conjunto vacio."""
    w, stack = _ventana_con_capas(qtbot)
    w._activate_tool("esfera")
    assert w._crop_dock is not None and not w._crop_dock.isHidden()
    w._on_layers_activated([])
    assert w._crop_dock.isHidden()


def test_una_nube_suelta_ya_aparece_como_capa_al_cargarla(app, qtbot, tmp_path):
    """Regresion: el panel de capas no salia hasta abrir una herramienta, y
    'En pantalla' decia que no habia nube con la nube cargada delante."""
    from app.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    rng = np.random.default_rng(0)
    pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(rng.uniform(0, 5, (500, 3))))
    ruta = tmp_path / "compuertas.ply"
    o3d.io.write_point_cloud(str(ruta), pcd)

    w._on_cloud_loaded(pcd, str(ruta))

    # Hay stack, sin necesidad de activar ninguna herramienta.
    assert w._active_tool is None
    assert w._layer_stack is not None and len(w._layer_stack) == 1
    assert w._layer_stack.active_indices == [0]

    # El panel de capas existe, esta visible y poblado, con el nombre del archivo.
    dock = w._ensure_layer_dock()
    assert not dock.isHidden()
    assert dock._n_filas() == 1
    from app.gui.panels.layer_dock import COL_NOMBRE
    assert "compuertas" in dock._item(0).text(COL_NOMBRE)

    # Y el resumen deja de decir que no hay nube.
    texto = w.info_panel._etiqueta_vista.text()
    assert "Sin nube cargada" not in texto
    assert "1 de 1 capas" in texto and "500" in texto

    # La nube se dibuja como capa, no como el actor suelto de antes.
    assert "layer_0" in w.viewer.plotter.actors
    assert "cloud_lidar" not in w.viewer.plotter.actors


def test_la_capa_de_una_nube_suelta_responde_a_la_visibilidad(app, qtbot, tmp_path):
    """Si el panel aparece, sus casillas tienen que hacer algo."""
    from app.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.zeros((100, 3))))
    ruta = tmp_path / "x.ply"
    o3d.io.write_point_cloud(str(ruta), pcd)
    w._on_cloud_loaded(pcd, str(ruta))

    w._on_layer_visibility(0, False)
    assert "layer_0" not in w.viewer.plotter.actors
    w._on_layer_visibility(0, True)
    assert "layer_0" in w.viewer.plotter.actors


def test_el_panel_de_informacion_se_puede_cerrar_y_recuperar(app, qtbot):
    """Era el unico panel sin boton de cierre: iba en un divisor, no en un dock."""
    from PyQt6.QtWidgets import QMenu

    from app.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    assert not w._info_dock.isHidden()
    # Y el visor ocupa el centro, no una mitad del divisor.
    assert w.centralWidget() is w.viewer

    w._info_dock.close()
    assert w._info_dock.isHidden()

    menu = QMenu()
    w._poblar_menu_paneles(menu)
    accion = next(a for a in menu.actions() if a.text() == "Información")
    accion.trigger()
    assert not w._info_dock.isHidden()


def test_los_tres_paneles_aparecen_en_el_menu(app, qtbot):
    from PyQt6.QtWidgets import QMenu

    from app.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    w._ensure_layer_dock()
    w._ensure_crop_dock()
    menu = QMenu()
    w._poblar_menu_paneles(menu)
    textos = [a.text() for a in menu.actions() if a.text()]
    assert textos == ["Información", "Capas", "Herramienta"]


def test_abrir_una_malla_avisa_antes_de_cargarla(app, qtbot, tmp_path, monkeypatch):
    """Regresion: la superficie se descartaba en silencio."""
    from PyQt6.QtWidgets import QMessageBox

    from app.gui import main_window as mw

    w = mw.MainWindow()
    qtbot.addWidget(w)
    malla = o3d.geometry.TriangleMesh.create_box(1, 1, 1)
    ruta = tmp_path / "cubo.ply"
    o3d.io.write_triangle_mesh(str(ruta), malla)

    monkeypatch.setattr(w, "_open_file_dialog", lambda *a, **k: str(ruta))
    monkeypatch.setattr(mw, "_e57_scan_count_safe", lambda p: 1)
    cargadas = []
    monkeypatch.setattr(w, "_cargar_nube_unica", lambda p: cargadas.append(p))

    # Si el usuario cancela, no se carga nada.
    preguntas = []

    def pregunta_no(*a, **k):
        preguntas.append(a[2] if len(a) > 2 else "")
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", pregunta_no)
    w._on_load_lidar()
    assert cargadas == []
    assert "12" in preguntas[0] and "vértices" in preguntas[0]

    # Si acepta, se carga como nube.
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)
    w._on_load_lidar()
    assert cargadas == [str(ruta)]


def test_abrir_una_nube_normal_no_pregunta_nada(app, qtbot, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from app.gui import main_window as mw

    w = mw.MainWindow()
    qtbot.addWidget(w)
    nube = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.zeros((100, 3))))
    ruta = tmp_path / "nube.ply"
    o3d.io.write_point_cloud(str(ruta), nube)

    monkeypatch.setattr(w, "_open_file_dialog", lambda *a, **k: str(ruta))
    monkeypatch.setattr(mw, "_e57_scan_count_safe", lambda p: 1)
    cargadas = []
    monkeypatch.setattr(w, "_cargar_nube_unica", lambda p: cargadas.append(p))
    preguntado = []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: preguntado.append(True))

    w._on_load_lidar()
    assert cargadas == [str(ruta)]
    assert preguntado == [], "una nube corriente no debe interrumpir al usuario"


def test_recachear_conserva_la_caja_al_desactivar_las_herramientas(
        app, qtbot, tmp_path, monkeypatch):
    """Regresion: `_deactivate_tools` ponia los limites a None ANTES de
    pasarlos al worker, y el recorte fallaba dentro del thread con un error de
    tipos incomprensible para el usuario."""
    from PyQt6.QtWidgets import QInputDialog, QMessageBox

    from app.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    w._e57_path = str(tmp_path / "x.e57")
    w._crop_min = np.array([0.0, 0.0, 0.0])
    w._crop_max = np.array([5.0, 6.0, 7.0])

    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (0.001, True))
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)
    creados = {}

    class WorkerFalso:
        def __init__(self, path, cache_dir, min_bound, max_bound, **kw):
            creados["min"] = min_bound
            creados["max"] = max_bound
            creados["voxel"] = kw.get("voxel_fino")
        progress = status = finished = error = None
        def start(self): creados["arrancado"] = True

    import app.core.workers as workers
    monkeypatch.setattr(workers, "RecacheWorker",
                        lambda *a, **k: _conectable(WorkerFalso(*a, **k)))
    w._on_recachear()

    assert creados["min"] is not None and creados["max"] is not None
    assert np.allclose(creados["min"], [0, 0, 0])
    assert np.allclose(creados["max"], [5, 6, 7])
    assert creados["voxel"] == 0.001
    # Y las herramientas si quedaron desactivadas.
    assert w._active_tool is None


class _Senal:
    def connect(self, *a, **k): pass


def _conectable(obj):
    """Da a un doble las senales que MainWindow conecta."""
    for nombre in ("progress", "status", "finished", "error"):
        setattr(obj, nombre, _Senal())
    return obj


def test_recortar_con_un_limite_vacio_da_un_mensaje_entendible():
    from app.core.io import _leer_e57_a_cache
    import pytest as _pytest

    with _pytest.raises(ValueError, match="límites"):
        _leer_e57_a_cache("x.e57", "cache", 0.01, 0.03, None,
                          bounds=(None, np.array([1.0, 1.0, 1.0])))
