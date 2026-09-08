"""
main_window.py — Ventana principal de Foster App.

Layout:
  ┌──────────────────────────────────────────────────┐
  │  Toolbar (cargar nube, vistas, recortar, segmentar)│
  ├───────────────────────┬──────────────────────────┤
  │                       │                          │
  │    Viewer 3D          │   Panel de información   │
  │    (PyVista/VTK)      │   (puntos, dimensiones)  │
  │                       │                          │
  ├───────────────────────┴──────────────────────────┤
  │  Barra de estado (mensajes + progreso)           │
  └──────────────────────────────────────────────────┘
"""

from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout,
    QToolBar, QStatusBar,
    QProgressBar, QLabel, QFileDialog, QDockWidget,
    QMessageBox, QApplication, QInputDialog,
    QMenu, QToolButton, QSlider, QVBoxLayout, QWidgetAction,
)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence

from app.core.layers import LayerStack, _slug
from app.core.project import Project
from app.core.workers import LoadWorker
from app.gui.viewer import Viewer3D
from app.gui.panels.crop_dock import CropDock
from app.gui.panels.layer_dock import LayerDock
from app.gui.panels.info_panel import InfoPanel


def _e57_total_puntos(path: str | None) -> int | None:
    """Puntos que contiene el archivo, leídos de las cabeceras.

    Se lee del encabezado y no de los datos: es instantáneo aunque el archivo
    pese decenas de gigas.
    """
    if not path or not path.lower().endswith(".e57"):
        return None
    try:
        import pye57
        e = pye57.E57(path)
        return sum(e.get_header(i).point_count for i in range(e.scan_count))
    except Exception:
        return None


def _e57_scan_count_safe(path: str) -> int:
    """Nº de scans de un .e57 (1 si no es .e57 o si falla la lectura del header)."""
    if not path.lower().endswith(".e57"):
        return 1
    try:
        from app.core.io import _e57_scan_count
        return _e57_scan_count(path)
    except Exception:
        return 1


class MainWindow(QMainWindow):
    """Ventana principal de Foster App."""

    APP_NAME = "Foster App"
    VERSION  = "0.1.0"

    def __init__(self):
        super().__init__()
        self.project = Project()
        self.settings = QSettings("LabPatrimonio", "FosterApp")
        self._crop_dock: CropDock | None = None
        self._layer_dock: LayerDock | None = None
        self._layer_stack: LayerStack | None = None
        self._active_tool: str | None = None   # None | "caja" | "lazo"
        self._crop_min: np.ndarray | None = None
        self._crop_max: np.ndarray | None = None
        self._lasso_mask: np.ndarray | None = None
        self._lasso_ops: list = []   # [(verts, set_op)] para reconstruir el recorte fino
        # Estado de la primitiva esfera
        self._esfera = None                 # Esfera ajustada (o None)
        self._esfera_mask = None            # máscara de lo capturado
        self._esfera_tol: float = 0.08
        self._esfera_phi: tuple[float, float] = (0.0, 180.0)
        # Estado de la primitiva plano
        self._plano = None
        self._plano_base = None      # el plano tal como lo dejó el ajuste
        self._plano_mask = None
        self._plano_offset: float = 0.0
        self._plano_tol: float = 0.08
        self._plano_radio: float = 5.0
        self._plano_pos = None       # (punto, normal) del widget, para re-ajustar
        # Estado de la primitiva cilindro
        self._cilindro = None
        self._cilindro_mask = None
        self._cilindro_tol: float = 0.08
        self._cilindro_h: tuple[float, float] = (-np.inf, np.inf)
        self._cilindro_theta: tuple[float, float] = (0.0, 360.0)
        # Estado de la primitiva cono
        self._cono = None
        self._cono_mask = None
        self._cono_tol: float = 0.08
        self._cono_h: tuple[float, float] = (-np.inf, np.inf)
        self._cono_theta: tuple[float, float] = (0.0, 360.0)
        self._cono_base = None          # el cono tal como lo dejó el ajuste
        self._cono_apertura: float = 0.0
        # Estado de DBSCAN
        self._db_etiquetas = None        # etiqueta de cluster por punto
        self._db_etiquetas_filtradas = None
        self._db_seleccion: list[int] = []
        self._db_umbral: int = 0
        # Estado de la reconstrucción
        self._malla = None
        self._malla_worker = None
        # Estado de la reparación por simetría
        self._si_plano = None
        self._si_relleno = None
        self._si_tol: float = 0.05
        self._si_zona: bool = False
        self._lasso_set_op: str = "union"
        # Origen de los scans, para poder re-cachear a otra resolución
        self._e57_path: str | None = None
        self._setup_window()
        self._setup_central()          # aquí se crea self.viewer
        self._setup_toolbar()
        self._setup_statusbar()
        self._apply_stylesheet()

        self.setWindowTitle(f"{self.APP_NAME} v{self.VERSION} · Sin título")
        self.show_status("Listo. Carga una nube de puntos para comenzar.")

    # ------------------------------------------------------------------ #
    # Setup UI                                                             #
    # ------------------------------------------------------------------ #

    def _setup_window(self):
        self.resize(1280, 800)
        # Restaurar geometría de la última sesión
        geometry = self.settings.value("window/geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def _setup_toolbar(self):
        tb = QToolBar("Principal")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(tb)

        # --- Cargar nube ---
        self._act_load_lidar = QAction("Cargar nube de puntos", self)
        self._act_load_lidar.setShortcut(QKeySequence("Ctrl+L"))
        self._act_load_lidar.setToolTip(
            "Cargar nube de puntos LiDAR (.e57, .las, .laz, .ply)\n"
            "Atajo: Ctrl+L"
        )
        self._act_load_lidar.triggered.connect(self._on_load_lidar)
        tb.addAction(self._act_load_lidar)

        # Va junto a la carga, no en Recortar: no recorta nada, vuelve a leer el
        # archivo original con otra resolución. Queda deshabilitada hasta que
        # haya una caja, porque necesita saber qué región releer.
        self._act_recachear = QAction("Re-cachear a resolución fina…", self)
        self._act_recachear.setToolTip(
            "Vuelve a leer el .e57 quedándose solo con la caja de recorte "
            "actual, usando un vóxel más fino. Requiere una caja definida y "
            "tarda varios minutos."
        )
        self._act_recachear.setEnabled(False)
        self._act_recachear.triggered.connect(self._on_recachear)
        tb.addAction(self._act_recachear)


        tb.addSeparator()

        tb.addWidget(self._menu_paneles())
        tb.addWidget(self._menu_vistas())
        tb.addWidget(self._menu_camara())
        tb.addWidget(self._menu_visibilidad())

        tb.addSeparator()

        # --- Recortar nube (menú desplegable Caja / Lazo) ---
        self._btn_crop = QToolButton()
        self._btn_crop.setText("Recortar")
        self._btn_crop.setToolTip("Herramientas de recorte de la nube")
        self._btn_crop.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._btn_crop.setEnabled(False)

        menu_crop = QMenu(self._btn_crop)
        self._nota(menu_crop,
                   "Quita lo que sobra. La capa se divide en lo conservado y "
                   "lo descartado; nada se pierde hasta que elimines una capa.")
        # Un solo grupo para los DOS menús: las herramientas siguen siendo
        # mutuamente excluyentes aunque estén repartidas en dos botones.
        grupo = QActionGroup(self)
        grupo.setExclusionPolicy(QActionGroup.ExclusionPolicy.ExclusiveOptional)

        def _accion(etiqueta: str, clave: str) -> QAction:
            act = QAction(etiqueta, self, checkable=True)
            act.setData(clave)
            grupo.addAction(act)
            act.triggered.connect(self._on_tool_action)
            return act

        # Recortar: quitar lo que sobra de la nube
        self._act_tool_caja = _accion("Caja", "caja")
        self._act_tool_lazo = _accion("Lazo", "lazo")
        for act in (self._act_tool_caja, self._act_tool_lazo):
            menu_crop.addAction(act)

        self._btn_crop.setMenu(menu_crop)
        tb.addWidget(self._btn_crop)

        # Segmentar: extraer entidades de lo que queda
        self._btn_seg = QToolButton()
        self._btn_seg.setText("Segmentar")
        self._btn_seg.setToolTip(
            "Primitivas que se ajustan a los datos y capturan una entidad")
        self._btn_seg.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._btn_seg.setEnabled(False)

        menu_seg = QMenu(self._btn_seg)
        self._nota(menu_seg,
                   "Extrae una entidad. Las primitivas ajustan una forma "
                   "conocida a los datos; agrupar y reparar operan sobre lo que "
                   "queda. Cada herramienta se explica en el panel lateral.")
        self._act_tool_esfera = _accion("Esfera", "esfera")
        self._act_tool_plano = _accion("Plano", "plano")
        self._act_tool_cilindro = _accion("Cilindro", "cilindro")
        self._act_tool_cono = _accion("Cono", "cono")
        self._act_tool_dbscan = _accion("Agrupar (DBSCAN)", "dbscan")
        self._act_tool_simetria = _accion("Reparar por simetría", "simetria")
        self._act_tool_malla = _accion("Mallar", "malla")
        for act in (self._act_tool_esfera, self._act_tool_plano,
                    self._act_tool_cilindro, self._act_tool_cono):
            menu_seg.addAction(act)
        menu_seg.addSeparator()
        menu_seg.addAction(self._act_tool_dbscan)
        menu_seg.addAction(self._act_tool_simetria)
        self._btn_seg.setMenu(menu_seg)
        tb.addWidget(self._btn_seg)

        # Mallar tiene botón propio y no vive dentro de Segmentar: no extrae
        # ninguna entidad, sino que convierte una ya extraída en superficie. Es
        # el paso siguiente del flujo, no una variante del anterior.
        self._btn_malla = QToolButton()
        self._btn_malla.setText("Mallar")
        self._btn_malla.setToolTip(
            "Convierte la entidad en superficie triangulada exportable.\n"
            "Último paso: sobre una entidad ya segmentada y reparada.")
        self._btn_malla.setDefaultAction(self._act_tool_malla)
        # Con una acción por defecto, el botón hereda su estado: habilitarlo o
        # no se decide en la acción, no en el botón.
        self._act_tool_malla.setEnabled(False)
        tb.addWidget(self._btn_malla)

        self._acts_tool = (self._act_tool_caja, self._act_tool_lazo,
                           self._act_tool_esfera, self._act_tool_plano,
                           self._act_tool_cilindro, self._act_tool_cono,
                           self._act_tool_dbscan, self._act_tool_simetria,
                           self._act_tool_malla)

        tb.addSeparator()

        # --- Limpiar escena ---
        act_clear = QAction("Limpiar", self)
        act_clear.setToolTip("Limpiar escena y proyecto actual")
        act_clear.triggered.connect(self._on_clear)
        tb.addAction(act_clear)

    def _habilitar_herramientas(self, activo: bool) -> None:
        """Los tres dependen de lo mismo: que haya nube cargada."""
        self._btn_crop.setEnabled(bool(activo))
        self._btn_seg.setEnabled(bool(activo))
        self._act_tool_malla.setEnabled(bool(activo))

    @staticmethod
    def _nota(menu: QMenu, texto: str) -> None:
        """Añade una línea explicativa dentro de un menú.

        Los tooltips solo aparecen al posarse encima y solo uno a la vez; para
        opciones cuyo funcionamiento no es evidente por el nombre, la
        explicación tiene que estar a la vista junto a la opción.
        """
        etiqueta = QLabel(texto)
        etiqueta.setWordWrap(True)
        etiqueta.setContentsMargins(28, 0, 12, 6)
        etiqueta.setStyleSheet("color: #999999; font-size: 11px;")
        etiqueta.setMaximumWidth(300)
        accion = QWidgetAction(menu)
        accion.setDefaultWidget(etiqueta)
        menu.addAction(accion)

    def _menu_paneles(self) -> QToolButton:
        """Vuelve a abrir un panel cerrado con su X.

        Sin esto, cerrar el dock de capas lo perdía para toda la sesión: los
        docks son cerrables y no había ningún sitio desde donde recuperarlos.
        El menú se puebla al desplegarse porque los docks se crean tarde —el de
        herramienta, solo cuando se usa una.
        """
        btn = QToolButton()
        btn.setText("Paneles")
        btn.setToolTip("Muestra u oculta los paneles laterales.")
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(btn)
        menu.aboutToShow.connect(lambda: self._poblar_menu_paneles(menu))
        btn.setMenu(menu)
        return btn

    def _poblar_menu_paneles(self, menu: QMenu) -> None:
        menu.clear()
        self._nota(menu, "Cerrar un panel con su X no lo destruye: se vuelve a "
                         "abrir desde aquí, con su contenido intacto.")
        hay = False
        for atributo, etiqueta in (("_info_dock", "Información"),
                                   ("_layer_dock", "Capas"),
                                   ("_crop_dock", "Herramienta")):
            dock = getattr(self, atributo, None)
            if dock is None:        # no se crea aquí: crearlo sería mostrarlo
                continue
            act = dock.toggleViewAction()
            act.setText(etiqueta)
            menu.addAction(act)
            hay = True
        if not hay:
            act = QAction("(no hay paneles todavía)", self)
            act.setEnabled(False)
            menu.addAction(act)

    def _menu_vistas(self) -> QToolButton:
        """Menú Vista: las seis caras del cubo más la isométrica.

        En una estructura de revolución como la cúpula, mirar desde un lado u
        otro cambia por completo qué queda ocluido, así que las tres vistas
        iniciales resultaban insuficientes.
        """
        btn = QToolButton()
        btn.setText("Vista")
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(btn)

        self._nota(menu, "Coloca la cámara en una vista fija. El zoom y el "
                         "desplazamiento se conservan.")
        self._acts_vista = []
        for i, (clave, etiqueta, _m, _n) in enumerate(self.viewer.VISTAS):
            act = QAction(etiqueta, self)
            act.triggered.connect(lambda _=False, c=clave: self.viewer.set_view(c))
            menu.addAction(act)
            self._acts_vista.append(act)
            if i == 0:              # separa la isométrica de las caras
                menu.addSeparator()
        # Cambiar la vista con un lazo a medio dibujar mezclaría vértices de
        # cámaras distintas: se bloquean mientras el lazo está activo.
        self._acts_vista = tuple(self._acts_vista)
        btn.setMenu(menu)
        return btn

    def _menu_camara(self) -> QToolButton:
        """Menú Cámara: modos de navegación, ambos apagados por defecto.

        Se dejan desactivados de entrada porque la órbita es lo que el usuario
        espera al abrir un visor 3D; activarlos es una decisión suya, no algo
        que tenga que deshacer.
        """
        btn = QToolButton()
        btn.setText("Cámara")
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(btn)

        self._act_fps = QAction("Vista libre", self, checkable=True)
        self._act_fps.setToolTip(
            "Manteniendo el botón derecho, mover el mouse gira la cámara como en "
            "un juego en primera persona. Al soltarlo recuperas el cursor para "
            "agarrar los manipuladores. Activa también WASD.")
        self._act_fps.toggled.connect(self._on_fps_toggled)
        menu.addAction(self._act_fps)
        self._nota(menu,
                   "Mantén el BOTÓN DERECHO y mueve el mouse para mirar, como "
                   "en un juego en primera persona. Al soltarlo recuperas el "
                   "cursor. Activa también WASD.")

        self._act_wasd = QAction("Navegación con teclado (WASD)", self, checkable=True)
        self._act_wasd.setToolTip(
            "W/S avanzar y retroceder, A/D lateral, E/Q subir y bajar. "
            "Con Shift el paso es 4× más grande. El paso se adapta a la "
            "distancia: cerca de la superficie es fino.")
        self._act_wasd.toggled.connect(self._on_wasd_toggled)
        menu.addAction(self._act_wasd)
        self._nota(menu,
                   "W/S avanzar y retroceder · A/D lateral · E/Q subir y bajar. "
                   "Con Shift el paso es 4× mayor. El paso se adapta a la "
                   "distancia: cerca de la superficie es fino.")

        menu.addSeparator()
        self._nota(menu,
                   "Ratón: arrastrar desplaza la nube · botón derecho gira · "
                   "Shift+arrastrar gira · rueda hace zoom.")


        btn.setMenu(menu)
        return btn

    def _menu_visibilidad(self) -> QToolButton:
        """Menú Visibilidad: opacidad y tamaño de punto.

        Los sliders van dentro del menú con QWidgetAction, no como diálogo: hay
        que poder arrastrarlos viendo el efecto en la nube, y un diálogo modal
        taparía justo lo que se está ajustando.
        """
        btn = QToolButton()
        btn.setText("Visibilidad")
        btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(btn)

        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(10, 8, 10, 8)

        intro = QLabel("Baja la opacidad o el tamaño de punto para ver los "
                       "manipuladores de una primitiva cuando quedan dentro de "
                       "la estructura.")
        intro.setWordWrap(True)
        intro.setMaximumWidth(280)
        intro.setStyleSheet("color: #999999; font-size: 11px;")
        lay.addWidget(intro)

        self._lbl_opacidad = QLabel()
        self._sld_opacidad = QSlider(Qt.Orientation.Horizontal)
        self._sld_opacidad.setRange(5, 100)
        self._sld_opacidad.setValue(100)
        self._sld_opacidad.setMinimumWidth(180)
        self._sld_opacidad.setToolTip(
            "Transparencia de la nube. Bájala para ver los manipuladores de una "
            "primitiva cuando quedan dentro de la estructura.")
        self._sld_opacidad.valueChanged.connect(self._on_opacidad)
        lay.addWidget(self._lbl_opacidad)
        lay.addWidget(self._sld_opacidad)

        self._lbl_punto = QLabel()
        self._sld_punto = QSlider(Qt.Orientation.Horizontal)
        self._sld_punto.setRange(1, 8)
        self._sld_punto.setValue(2)
        self._sld_punto.setMinimumWidth(180)
        self._sld_punto.setToolTip(
            "Tamaño del punto en píxeles. Puntos chicos dejan huecos entre sí y "
            "hacen visible la geometría de detrás.")
        self._sld_punto.valueChanged.connect(self._on_tamano_punto)
        lay.addWidget(self._lbl_punto)
        lay.addWidget(self._sld_punto)

        self._actualizar_labels_visibilidad()
        accion = QWidgetAction(menu)
        accion.setDefaultWidget(panel)
        menu.addAction(accion)
        btn.setMenu(menu)
        return btn

    def _actualizar_labels_visibilidad(self) -> None:
        self._lbl_opacidad.setText(f"Opacidad de la nube: {self._sld_opacidad.value()} %")
        self._lbl_punto.setText(f"Tamaño de punto: {self._sld_punto.value()} px")

    def _on_opacidad(self, v: int) -> None:
        self._actualizar_labels_visibilidad()
        self.viewer.set_opacidad_nube(v / 100.0)

    def _on_tamano_punto(self, v: int) -> None:
        self._actualizar_labels_visibilidad()
        self.viewer.set_tamano_punto(float(v))

    def _setup_central(self):
        """El visor ocupa el centro; la información va en un dock lateral.

        Estaba incrustada en un divisor junto al visor, y por eso era el único
        panel que no se podía cerrar ni recuperar: los divisores no tienen botón
        de cierre y el menú Paneles solo conoce docks. Como dock se comporta
        igual que Capas y Herramienta, que es lo que el usuario espera al ver
        tres paneles iguales en pantalla.
        """
        self.viewer = Viewer3D(self)
        self.viewer.points_displayed.connect(self._on_points_displayed)
        self.setCentralWidget(self.viewer)

        self.info_panel = InfoPanel(self)
        self._info_dock = QDockWidget("Información", self)
        self._info_dock.setObjectName("info_dock")
        self._info_dock.setWidget(self.info_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._info_dock)

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        self._status_label = QLabel("Listo")
        self.statusbar.addWidget(self._status_label, 1)

        self._progress = QProgressBar()
        self._progress.setMaximumWidth(200)
        self._progress.setVisible(False)
        self.statusbar.addPermanentWidget(self._progress)

        self._points_label = QLabel("")
        self.statusbar.addPermanentWidget(self._points_label)

    def _apply_stylesheet(self):
        """Tema oscuro para toda la aplicación."""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #2b2b2b;
                color: #dddddd;
            }
            QToolBar {
                background-color: #3c3c3c;
                border-bottom: 1px solid #555555;
                padding: 4px;
                spacing: 4px;
            }
            QToolButton {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 4px 8px;
                color: #dddddd;
            }
            QToolButton:hover {
                background-color: #4a4a4a;
                border: 1px solid #666666;
            }
            QToolButton:pressed {
                background-color: #555555;
            }
            QGroupBox {
                border: 1px solid #555555;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                color: #aaaaaa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: #aaaaaa;
            }
            QStatusBar {
                background-color: #1e1e1e;
                border-top: 1px solid #444444;
            }
            QLabel {
                color: #dddddd;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                background-color: #3c3c3c;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2e75b6;
                border-radius: 2px;
            }
            QPushButton {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #777777;
            }
            QPushButton:pressed {
                background-color: #2e75b6;
            }
            QPushButton:disabled {
                color: #777777;
                border-color: #444444;
            }
            QRadioButton {
                spacing: 6px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
                border-radius: 8px;
                border: 2px solid #888888;
                background-color: #2b2b2b;
            }
            QRadioButton::indicator:hover {
                border-color: #cccccc;
            }
            QRadioButton::indicator:checked {
                background-color: #4caf50;
                border-color: #dddddd;
            }
        """)

    # ------------------------------------------------------------------ #
    # Acciones de la toolbar                                               #
    # ------------------------------------------------------------------ #

    def _on_load_lidar(self):
        """Carga una nube: un .e57 con varios scans se abre como capas por scan;
        cualquier otro archivo (o .e57 de un scan) como nube única."""
        path = self._open_file_dialog("Cargar nube de puntos")
        if not path:
            return
        if not self._confirmar_si_es_malla(path):
            return
        if _e57_scan_count_safe(path) >= 2:
            self._cargar_scans_por_capas(path)
        else:
            self._cargar_nube_unica(path)

    def _confirmar_si_es_malla(self, path: str) -> bool:
        """Avisa si el archivo trae una malla, porque solo se cargan sus vértices.

        Sin este aviso la pérdida es silenciosa. El diálogo de apertura acepta
        `.ply`, y si el archivo contiene una malla exportada por la propia
        herramienta el visor muestra una nube dispersa de vértices sin que nada
        indique que la superficie se quedó fuera.
        """
        from app.core.io import caras_en_ply

        caras, vertices = caras_en_ply(path)
        if caras <= 0:
            return True
        respuesta = QMessageBox.question(
            self,
            "El archivo contiene una malla",
            f"'{Path(path).name}' declara {caras:,} caras y {vertices:,} "
            "vértices.\n\n"
            "Esta herramienta trabaja con nubes de puntos, así que cargará "
            "únicamente los vértices y descartará la superficie.\n\n"
            "¿Quieres continuar?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        return respuesta == QMessageBox.StandardButton.Yes

    def _cargar_nube_unica(self, path: str) -> None:
        self._load_cloud(path)

    @staticmethod
    def _espaciado_de(pcd):
        """Distancia típica entre puntos vecinos. Es el dato que gobierna el
        vóxel del caché, el eps de la agrupación y la tolerancia de las
        primitivas; mostrarlo evita que el usuario lo deduzca probando."""
        try:
            from app.modules.clusters import espaciado_medio
            pts = np.asarray(pcd.points)
            return espaciado_medio(pts) if len(pts) > 1 else None
        except Exception:
            return None

    def _cargar_scans_por_capas(self, path: str) -> None:
        """Carga el .e57 como una capa por scan (con caché de dos resoluciones)."""
        cache_dir = Path("output/scans_cache") / Path(path).stem
        self._e57_path = path      # se necesita para re-cachear a resolución fina
        self.show_status("Cargando scans (1ª vez puede tardar; luego usa caché)...")
        self._set_loading(True)

        from app.core.workers import ScanLayersWorker
        worker = ScanLayersWorker(path, cache_dir, parent=self)
        worker.progress.connect(self._progress.setValue)
        worker.status.connect(self.show_status)
        worker.finished.connect(self._on_scan_layers_loaded)
        worker.error.connect(self._on_load_error)
        self._current_worker = worker
        worker.start()

    def _on_scan_layers_loaded(self, scans) -> None:
        """Puebla el LayerStack con una capa por scan (versión gruesa + fine_path)."""
        from app.core.layers import CloudLayer, LayerStack
        self._set_loading(False)
        stack = LayerStack()
        stack.layers = [
            CloudLayer(name=f"Scan {s.index:02d}", pcd=s.pcd_grueso,
                       fine_path=s.fine_path, n_pts_archivo=s.n_pts_fino)
            for s in scans
        ]
        stack.active_index = 0
        self._layer_stack = stack
        # El panel de información solo se actualizaba en la carga de nube única;
        # con la carga por capas —que es el camino habitual— quedaba en blanco.
        self._actualizar_info_de_capas(scans)
        self._redibujar_capas(stack)
        # Solo el dock de capas: el de herramientas aparece al activar una.
        self._ensure_layer_dock().refresh_layers(stack)
        self._habilitar_herramientas(True)
        self.setWindowTitle(f"{self.APP_NAME} v{self.VERSION} · scans")
        self.show_status(f"{len(scans)} scans cargados como capas. "
                         "Apaga los interiores y usa 'Exportar visibles'.")

    def _actualizar_info_de_capas(self, scans) -> None:
        """Refleja en el panel la nube cargada por capas."""
        from app.core.project import CloudInfo

        pts = np.vstack([np.asarray(s.pcd_grueso.points) for s in scans])
        en_cache = sum(s.n_pts_fino for s in scans)
        # El caché guarda los puntos YA voxelizados: informar ese número como si
        # fuera el del archivo da una idea equivocada del dato de partida.
        del_archivo = _e57_total_puntos(self._e57_path) or en_cache
        self.project.lidar_info = CloudInfo(
            name=Path(self._e57_path).stem if self._e57_path else "escaneos",
            path=self._e57_path or "",
            n_points_original=del_archivo,
            n_points_display=len(pts),
            has_colors=bool(scans and scans[0].pcd_grueso.has_colors()),
            bbox_min=pts.min(axis=0), bbox_max=pts.max(axis=0),
        )
        self.info_panel.set_scan_count(len(scans))
        self.info_panel.set_puntos_en_cache(en_cache)
        self.info_panel.set_espaciado(self._espaciado_de(scans[0].pcd_grueso))
        self.info_panel.update_from_project(self.project)

    def _on_clear(self):
        """Limpia la escena y reinicia el proyecto."""
        reply = QMessageBox.question(
            self,
            "Limpiar proyecto",
            "¿Limpiar toda la escena? Esta acción no se puede deshacer.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._deactivate_tools()
            if self._crop_dock is not None:
                self._crop_dock.hide()
            self._layer_stack = None
            self.viewer.clear()
            self.project.clear()
            self.info_panel.update_from_project(self.project)
            self._points_label.setText("")
            self._habilitar_herramientas(False)
            self.setWindowTitle(f"{self.APP_NAME} v{self.VERSION} · Sin título")
            self.show_status("Escena limpiada.")

    def _open_file_dialog(self, title: str) -> str:
        """Abre el diálogo de archivo y retorna la ruta seleccionada."""
        from app.core.io import FILTER_STRING
        last_dir = self.settings.value("io/last_dir", os.path.expanduser("~"))

        path, _ = QFileDialog.getOpenFileName(
            self, title, last_dir, FILTER_STRING
        )

        if path:
            self.settings.setValue("io/last_dir", os.path.dirname(path))

        return path

    # ------------------------------------------------------------------ #
    # Carga de nubes (en background)                                       #
    # ------------------------------------------------------------------ #

    def _load_cloud(self, path: str):
        """Lanza el worker de carga en background."""
        self.show_status("Cargando nube de puntos...")
        self._set_loading(True)

        worker = LoadWorker(path, parent=self)
        worker.progress.connect(self._progress.setValue)
        worker.finished.connect(self._on_cloud_loaded)
        worker.error.connect(self._on_load_error)

        # Guardar referencia para evitar que el GC destruya el worker
        self._current_worker = worker
        worker.start()

    def _on_cloud_loaded(self, pcd, path: str):
        """Callback cuando el worker termina de cargar la nube."""
        self._set_loading(False)
        self.project.set_lidar(pcd, path)
        self.viewer.show_cloud(pcd, name="cloud_lidar", point_size=2.0)

        self.info_panel.set_scan_count(_e57_scan_count_safe(path)
                                       if path.lower().endswith(".e57") else None)
        self.info_panel.set_espaciado(self._espaciado_de(pcd))
        self.info_panel.update_from_project(self.project)
        self.setWindowTitle(
            f"{self.APP_NAME} v{self.VERSION} · {self.project.name}"
        )
        self.show_status(f"Cargado: {path}")
        self._habilitar_herramientas(True)

        # El stack se crea AL CARGAR y no al activar la primera herramienta.
        # Aplazarlo tenía dos efectos visibles: el panel de capas no aparecía
        # hasta abrir una herramienta, y el resumen "En pantalla" seguía diciendo
        # que no había nube con la nube delante, porque se calcula desde el
        # stack. Una nube suelta es una capa como cualquier otra.
        stack = LayerStack()
        stack.reset(pcd, name=Path(path).stem or "Original")
        self._layer_stack = stack
        dock = self._ensure_layer_dock()
        dock.refresh_layers(stack)
        dock.show()
        self._repartir_docks()
        self._redibujar_capas(stack)

    def _on_load_error(self, message: str):
        """Callback cuando la carga falla."""
        self._set_loading(False)
        self.show_status(f"Error al cargar archivo.")
        QMessageBox.critical(
            self,
            "Error al cargar archivo",
            f"No se pudo cargar el archivo:\n\n{message}",
        )

    def _on_points_displayed(self, n_original: int, n_displayed: int):
        """Actualiza el contador de puntos en la barra de estado."""
        if n_original == n_displayed:
            self._points_label.setText(f"  {n_original:,} puntos")
        else:
            self._points_label.setText(
                f"  {n_original:,} puntos "
                f"(mostrando {n_displayed:,})"
            )

    # ------------------------------------------------------------------ #
    # Helpers UI                                                           #
    # ------------------------------------------------------------------ #

    def show_status(self, message: str):
        self._status_label.setText(message)

    def _set_loading(self, loading: bool):
        self._progress.setVisible(loading)
        self._act_load_lidar.setEnabled(not loading)

        if loading:
            self._progress.setValue(0)
        QApplication.processEvents()

    # ------------------------------------------------------------------ #
    # Recorte interactivo                                                  #
    # ------------------------------------------------------------------ #

    def _ensure_layer_dock(self) -> LayerDock:
        """Crea el dock de capas la primera vez. Existe desde que hay nube, a
        diferencia del de herramientas, que solo aparece al usar una."""
        if self._layer_dock is None:
            dock = LayerDock(self)
            dock.layer_visibility_changed.connect(self._on_layer_visibility)
            # El conjunto activo lo dice la columna 'Usar', no la selección.
            dock.layers_activated.connect(self._on_layers_activated)
            dock.activate_visible_requested.connect(self._on_activate_visible)
            dock.layer_removed.connect(self._on_layer_removed)
            dock.layers_removed.connect(self._on_layers_removed)
            dock.layer_renamed.connect(self._on_layer_renamed)
            dock.layer_restore_requested.connect(self._on_layer_restore)
            dock.layers_merge_requested.connect(self._on_layers_merge)
            dock.layers_visibility_changed.connect(self._on_layers_visibility)
            dock.layers_isolate_requested.connect(self._on_layers_isolate)
            dock.layers_invert_requested.connect(self._on_layers_invert)
            dock.discards_removal_requested.connect(self._on_remove_discards)
            dock.export_requested.connect(self._on_export_visible)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
            self._layer_dock = dock
        return self._layer_dock

    def _repartir_docks(self) -> None:
        """Da a cada dock una altura razonable cuando ambos están visibles.

        Sin esto, Qt reparte el área a partes iguales y la lista de capas puede
        quedar en una sola fila.
        """
        if self._layer_dock is None or self._crop_dock is None:
            return
        if not self._crop_dock.isVisible():
            return
        alto = max(self.height(), 600)
        self.resizeDocks([self._layer_dock, self._crop_dock],
                         [int(alto * 0.45), int(alto * 0.55)],
                         Qt.Orientation.Vertical)

    def _ensure_crop_dock(self) -> CropDock:
        if self._crop_dock is None:
            dock = CropDock(self)
            dock.tool_apply.connect(self._on_box_apply)
            dock.tool_cancel.connect(self._on_tool_cancel)
            dock.esfera_params_changed.connect(self._on_esfera_params)
            dock.esfera_capturar.connect(self._on_esfera_capturar)
            dock.lasso_started.connect(self._on_lasso_started)
            dock.lasso_apply.connect(self._on_lasso_apply)
            dock.lasso_cancel.connect(self._on_lasso_cancel)
            dock.plano_params_changed.connect(self._on_plano_params)
            dock.plano_capturar.connect(self._on_plano_capturar)
            dock.plano_reajustar.connect(self._on_plano_reajustar)
            dock.cilindro_params_changed.connect(self._on_cilindro_params)
            dock.cilindro_capturar.connect(self._on_cilindro_capturar)
            dock.cono_params_changed.connect(self._on_cono_params)
            dock.cono_capturar.connect(self._on_cono_capturar)
            dock.cono_reajustar.connect(self._on_cono_reajustar)
            dock.dbscan_agrupar.connect(self._on_dbscan_agrupar)
            dock.dbscan_eps_changed.connect(self._on_dbscan_eps)
            dock.dbscan_umbral_changed.connect(self._on_dbscan_umbral)
            dock.dbscan_seleccion_changed.connect(self._on_dbscan_seleccion)
            dock.dbscan_capturar.connect(self._on_dbscan_capturar)
            dock.dbscan_eliminar.connect(self._on_dbscan_eliminar)
            dock.simetria_params_changed.connect(self._on_simetria_params)
            dock.simetria_refinar.connect(self._on_simetria_refinar)
            dock.malla_generar.connect(self._on_malla_generar)
            dock.malla_exportar.connect(self._on_malla_exportar)
            dock.malla_vista_changed.connect(self._on_malla_vista)
            dock.simetria_aplicar.connect(self._on_simetria_aplicar)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
            # Crear NO es mostrar: varias operaciones sobre capas necesitan el
            # dock para apagar el lazo, y si crearlo lo hiciera visible el panel
            # de herramientas aparecería sin herramienta activa, robándole
            # espacio a la lista de capas.
            dock.hide()
            self._crop_dock = dock
        return self._crop_dock

    def _ensure_layer_stack(self) -> bool:
        """Asegura que exista un stack de capas. Si ya hay uno (p. ej. cargado por
        scans), lo usa; si no, lo crea desde la nube única del project."""
        if self._layer_stack is not None:
            if not self._layer_stack.active_indices:
                self.show_status(
                    "Ninguna capa en uso: marca al menos una en la columna "
                    "'Usar' para elegir sobre qué trabaja la herramienta.")
                return False
            return True
        source = self.project.fused_cloud or self.project.lidar_cloud
        if source is None:
            return False
        self._layer_stack = LayerStack()
        self._layer_stack.reset(source)
        self._redibujar_capas(self._layer_stack)
        return True

    def _on_tool_action(self) -> None:
        """Handler de las acciones del menú Recortar (Caja/Lazo)."""
        accion = self.sender()
        if not accion.isChecked():
            self._deactivate_tools()
            self.show_status("Herramienta de recorte desactivada.")
            return
        self._activate_tool(accion.data())

    def _activate_tool(self, tool: str) -> None:
        if not self._ensure_layer_stack():
            for act in self._acts_tool:
                act.setChecked(False)
            return
        self._deactivate_tools(keep_checked=tool)
        dock = self._ensure_crop_dock()
        dock.set_tool(tool)
        self._ensure_layer_dock().refresh_layers(self._layer_stack)
        dock.show()
        dock.raise_()
        self._repartir_docks()
        self._active_tool = tool
        if tool == "caja":
            self.viewer.start_crop_widget(
                self._layer_stack.active.pcd,
                callback=self._on_crop_bounds_changed,
            )
            self.show_status("Ajusta la caja: VERDE se conserva, ROJO se elimina.")
        elif tool == "esfera":
            self._esfera = None
            dock.set_esfera_has_selection(False)
            self.viewer.start_sphere_widget(
                self._layer_stack.active.pcd,
                callback=self._on_esfera_movida,
            )
            self.show_status(
                "Arrastra la esfera sobre la nube. Al soltar se ajusta y se "
                "marca en VERDE lo que capturaría."
            )
        elif tool == "plano":
            self._plano = None
            dock.set_plano_has_selection(False)
            self._plano_pos = self.viewer.start_plane_widget(
                self._layer_stack.active.pcd,
                callback=self._on_plano_movido,
            )
            self.show_status(
                "Coloca el plano sobre la superficie. Ajusta el RADIO: un plano "
                "sin acotar captura todo lo coplanar de la nube."
            )
        elif tool == "cilindro":
            self._cilindro = None
            dock.set_cilindro_has_selection(False)
            self.viewer.start_line_widget(
                self._layer_stack.active.pcd,
                callback=self._on_cilindro_movido,
            )
            self.show_status(
                "Arrastra los extremos de la linea para dar el eje del cilindro. "
                "Al soltar se ajusta el radio a los datos."
            )
        elif tool == "cono":
            self._cono = None
            dock.set_cono_has_selection(False)
            self.viewer.start_line_widget(
                self._layer_stack.active.pcd,
                callback=self._on_cono_movido,
            )
            self.show_status(
                "Arrastra la linea para dar el eje del cono. Al soltar se ajusta "
                "la apertura a los datos."
            )
        elif tool == "dbscan":
            self._db_etiquetas = None
            self._db_seleccion = []
            dock.refresh_clusters([])
            self._mostrar_referencia_dbscan()
            self.show_status(
                "Ajusta eps y min_points, y pulsa 'Agrupar'. Es la operacion "
                "cara, por eso no se dispara al mover los sliders."
            )
        elif tool == "simetria":
            self._si_plano = self._si_relleno = None
            dock.set_simetria_concordancia(None)
            self.viewer.start_plane_widget(
                self._layer_stack.active.pcd,
                callback=self._on_simetria_plano_movido,
            )
            self.show_status(
                "Coloca el plano de simetria mas o menos donde va y pulsa "
                "'Refinar plano'. La concordancia dice si la simetria existe."
            )
        elif tool == "malla":
            self._malla = None
            dock.set_malla_resultado(None)
            self._mostrar_referencia_malla()
            self.show_status(
                "Marca en 'Usar' la entidad que quieras mallar y pulsa "
                "'Generar malla'. El limite de perimetro viene propuesto a "
                "partir del espaciado de la nube."
            )
        else:
            self.show_status(
                "Elige la operacion y pulsa 'Iniciar lazo' en el panel Recorte."
            )

    def _deactivate_tools(self, keep_checked: str | None = None) -> None:
        """Detiene caja y lazo, limpia previews y desmarca las acciones del menú."""
        for act in self._acts_vista:
            act.setEnabled(True)
        self.viewer.stop_crop_widget()
        self.viewer.stop_sphere_widget()
        self.viewer.stop_plane_widget()
        self.viewer.stop_line_widget()
        self.viewer.ocultar_fantasma_cono()
        self.viewer.ocultar_malla()   # devuelve las capas a la vista
        self.viewer.stop_lasso()
        self.viewer.clear_preview()
        self._lasso_mask = None
        self._lasso_ops = []
        self._esfera = None
        self._esfera_mask = None
        self._plano = self._plano_base = self._plano_mask = None
        self._cilindro = self._cilindro_mask = None
        self._cono = self._cono_base = self._cono_mask = None
        self._db_etiquetas = self._db_etiquetas_filtradas = None
        self._db_seleccion = []
        self._si_plano = self._si_relleno = None
        self.viewer.ocultar_clusters()
        self.viewer.ocultar_relleno()
        self._crop_min = None
        self._crop_max = None
        self._active_tool = None
        if self._crop_dock is not None:
            self._ensure_crop_dock().set_lasso_active(False)
            self._ensure_crop_dock().set_lasso_has_selection(False)
        for act in self._acts_tool:
            if act.data() != keep_checked:
                act.setChecked(False)

    _EDITS_DIR = Path("output/scans_cache/_edits")

    def _fine_keep_fn(self):
        """Criterio de recorte reaplicable a la nube fina, según la herramienta."""
        if self._active_tool == "caja" and self._crop_min is not None:
            mn, mx = self._crop_min, self._crop_max

            def fn(pcd_fino):
                pts = np.asarray(pcd_fino.points)
                return np.all((pts >= mn) & (pts <= mx), axis=1)
            return fn
        if self._active_tool == "lazo" and self._lasso_ops:
            from app.modules.processing import apply_lasso
            ops = list(self._lasso_ops)      # [(verts, set_op), ...] en orden

            def fn(pcd_fino):
                screen, valid = self.viewer.project_cloud_to_screen(pcd_fino)
                mask = np.zeros(len(screen), dtype=bool)
                for verts, op in ops:
                    base = mask
                    if not mask.any() and op in ("difference", "intersection"):
                        base = np.ones(len(screen), dtype=bool)
                    mask = apply_lasso(screen, valid, verts, op, base)
                return mask
            return fn
        return lambda pcd_fino: None    # sin criterio → split normal

    def _apply_split(self, keep_mask: np.ndarray) -> None:
        """Núcleo común de Aplicar (caja y lazo): divide la capa activa."""
        stack = self._layer_stack
        if stack is None or keep_mask is None:
            return
        n_activas = len(stack.active_indices)
        fuente_nombre = stack.activas[0].name
        try:
            n_div, n_comp, n_sin = stack.split_activas(
                keep_mask, self._fine_keep_fn(), self._EDITS_DIR
            )
        except ValueError as e:
            # Nada que separar (todo verde o todo rojo): limpiar el preview
            # para no dejar la capa pintada, y partir de cero.
            self.viewer.stop_lasso()
            self.viewer.clear_preview()
            self._lasso_mask = None
            dock = self._ensure_crop_dock()
            dock.set_lasso_active(False)
            dock.set_lasso_has_selection(False)
            self.show_status(str(e))
            return

        self._lasso_mask = None
        self._lasso_ops = []
        self.viewer.stop_lasso()
        self.viewer.clear_preview()
        self._redibujar_capas(stack)
        dock = self._ensure_crop_dock()
        self._ensure_layer_dock().refresh_layers(stack)
        dock.set_lasso_active(False)
        dock.set_lasso_has_selection(False)

        if self._active_tool == "caja":
            self.viewer.start_crop_widget(
                stack.active.pcd, callback=self._on_crop_bounds_changed
            )

        capturados = sum(len(c.pcd.points) for c in stack.activas)
        if n_activas == 1:
            self.show_status(
                f"'{fuente_nombre}' quedó dividida: {capturados:,} pts "
                f"capturados. El resto quedó oculto y marcado como descarte, "
                "listo para restaurar si te equivocaste.")
        else:
            self.show_status(
                f"{n_activas} capas activas: {n_div} divididas, {n_comp} "
                f"capturadas enteras, {n_sin} sin aporte. {capturados:,} pts "
                "capturados en total; cada capa conserva su nombre de origen.")

    def _on_fps_toggled(self, activo: bool) -> None:
        # mirar sin poder desplazarse no sirve de nada: van juntos
        if activo and not self._act_wasd.isChecked():
            self._act_wasd.setChecked(True)
        self.viewer.set_mouse_look(bool(activo))
        self.show_status(
            "Vista libre activada: MANTÉN EL BOTÓN DERECHO y mueve el mouse para "
            "mirar, W/A/S/D para desplazarte, E/Q para subir y bajar. Suelta el "
            "botón derecho para volver a agarrar los manipuladores."
            if activo else
            "Vista libre desactivada. Arrastra para girar alrededor del objeto.")

    def _on_wasd_toggled(self, activo: bool) -> None:
        self.viewer.wasd_activo = bool(activo)
        self.show_status(
            "Navegación WASD activada: W/S adelante-atrás, A/D lateral, E/Q "
            "subir-bajar (Shift = paso grande)."
            if activo else "Navegación WASD desactivada."
        )

    def _on_layer_renamed(self, i: int, nombre: str) -> None:
        """Renombrar es lo que convierte una capa en una entidad identificable:
        'Recorte 3' no dice nada, 'Cúpula exterior' sí."""
        if self._layer_stack is None:
            return
        try:
            self._layer_stack.renombrar(i, nombre)
        except (ValueError, IndexError) as e:
            self.show_status(str(e))
            return
        self._ensure_layer_dock().refresh_layers(self._layer_stack)
        self.show_status(f"Capa renombrada a '{nombre}'.")

    def _on_remove_discards(self) -> None:
        """Borra de una vez todo lo que quedó fuera de los recortes."""
        stack = self._layer_stack
        if stack is None:
            return
        n = stack.contar_descartes()
        if n == 0:
            self.show_status("No hay capas de descarte.")
            return
        resp = QMessageBox.question(
            self, "Eliminar descartes",
            f"Se eliminarán {n} capas marcadas como descarte.\n"
            f"Esto NO se puede deshacer con 'Restaurar eliminada'.\n\n¿Continuar?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        try:
            n = stack.eliminar_descartes()
        except ValueError as e:
            QMessageBox.warning(self, "Eliminar descartes", str(e))
            return
        self._deactivate_tools()
        self._redibujar_capas(stack)
        self._ensure_layer_dock().refresh_layers(stack)
        self.show_status(f"{n} capas de descarte eliminadas.")

    def _apply_split_visible(self) -> None:
        """Aplica la caja a todas las capas visibles de una vez (feature 0.a).

        Cada capa partida conserva su nombre de origen en las dos mitades: sin
        eso se pierde de qué scan viene cada punto, que es justamente lo que
        permite separar después el interior del exterior apagando capas.
        """
        stack = self._layer_stack
        if stack is None:
            return
        try:
            n_div, n_ocul, n_intacta = stack.split_visible_fino(
                self._box_keep_fn(), self._fine_keep_fn(), self._EDITS_DIR
            )
        except ValueError as e:
            self.viewer.clear_preview()
            self.show_status(str(e))
            return

        self.viewer.clear_preview()
        self._redibujar_capas(stack)
        dock = self._ensure_crop_dock()
        self._ensure_layer_dock().refresh_layers(stack)
        if self._active_tool == "caja":
            self.viewer.start_crop_widget(
                stack.active.pcd, callback=self._on_crop_bounds_changed
            )
        self.show_status(
            f"Recorte aplicado a las capas visibles: {n_div} divididas, "
            f"{n_intacta} sin cambios, {n_ocul} ocultas por quedar fuera."
        )

    # ---------- herramienta caja ---------- #

    def _box_keep_fn(self):
        """Criterio de la caja, aplicable a cualquier nube (grueso o fino)."""
        mn, mx = self._crop_min, self._crop_max

        def fn(pcd):
            pts = np.asarray(pcd.points)
            return np.all((pts >= mn) & (pts <= mx), axis=1)
        return fn

    def _capas_visibles(self):
        """Capas visibles del stack, con su índice (para nombrar los actores)."""
        stack = self._layer_stack
        return [(i, c) for i, c in enumerate(stack.layers) if c.visible]

    def _on_crop_bounds_changed(
        self, min_bound: np.ndarray, max_bound: np.ndarray
    ) -> None:
        """Callback del box widget: preview verde (dentro) / rojo (fuera)."""
        self._crop_min = min_bound
        self._crop_max = max_bound
        # el re-cacheo necesita una caja: recién ahora tiene sentido ofrecerlo
        self._act_recachear.setEnabled(self._e57_path is not None)
        stack = self._layer_stack
        if stack is None or self._crop_min is None:
            return

        # El preview cubre TODAS las capas visibles, no solo la activa: con 35
        # scans, ver el recorte de a uno no dice nada sobre lo que va a pasar.
        visibles = self._capas_visibles()
        if not visibles:
            return
        fn = self._box_keep_fn()
        pts = np.vstack([np.asarray(c.pcd.points) for _, c in visibles])
        keep = np.concatenate([fn(c.pcd) for _, c in visibles])
        combinada = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        self.viewer.preview_split(
            combinada, keep, [f"layer_{i}" for i, _ in visibles]
        )

    def _on_box_apply(self) -> None:
        if self._layer_stack is None or self._crop_min is None:
            return
        self._apply_split_visible()

    def _on_recachear(self) -> None:
        """
        Re-lee el .e57 recortado a la caja actual, con un vóxel más fino.

        El caché guarda los puntos YA voxelizados, así que el detalle perdido no
        se puede recuperar de un .ply local: hay que volver al archivo original.
        Por eso se avisa del costo antes de empezar.
        """
        if self._e57_path is None or self._crop_min is None:
            QMessageBox.information(
                self, "Re-cachear",
                "Primero carga un .e57 por capas y define una caja de recorte.")
            return

        voxel, ok = QInputDialog.getDouble(
            self, "Re-cachear a resolución fina",
            "Tamaño de vóxel en metros:\n\n"
            "Debe ser bastante menor que el espesor más fino que quieras\n"
            "distinguir; si no, dos caras cercanas se fusionan en una sola.",
            0.01, 0.001, 0.10, 3,
        )
        if not ok:
            return

        ext = np.asarray(self._crop_max) - np.asarray(self._crop_min)
        resp = QMessageBox.question(
            self, "Confirmar re-cacheo",
            f"Se volverá a leer el archivo completo, conservando solo la caja de\n"
            f"{ext[0]:.1f} × {ext[1]:.1f} × {ext[2]:.1f} m, con vóxel de {voxel:.3f} m.\n\n"
            f"Puede tardar varios minutos. El caché actual no se modifica.\n\n"
            f"¿Continuar?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return

        # Los límites se copian ANTES de desactivar las herramientas, porque
        # `_deactivate_tools` los pone a None y el worker los recibiría vacíos.
        min_bound = np.asarray(self._crop_min, dtype=float).copy()
        max_bound = np.asarray(self._crop_max, dtype=float).copy()

        self._deactivate_tools()
        cache_dir = (Path("output/scans_cache")
                     / f"{Path(self._e57_path).stem}_fino_{voxel:.3f}")
        self.show_status("Re-cacheando a resolución fina (esto tarda)...")
        self._set_loading(True)

        from app.core.workers import RecacheWorker
        worker = RecacheWorker(
            self._e57_path, cache_dir, min_bound, max_bound,
            voxel_fino=voxel, voxel_grueso=max(voxel * 3, 0.03), parent=self,
        )
        worker.progress.connect(self._progress.setValue)
        worker.status.connect(self.show_status)
        worker.finished.connect(self._on_recache_listo)
        worker.error.connect(self._on_load_error)
        self._current_worker = worker
        worker.start()

    def _on_recache_listo(self, scans) -> None:
        """Reemplaza el LayerStack por los scans re-cacheados."""
        self._crop_min = self._crop_max = None      # la caja ya se consumió
        self._act_recachear.setEnabled(False)
        self._on_scan_layers_loaded(scans)
        self.show_status(
            f"{len(scans)} scans re-cacheados a resolución fina dentro de la caja.")

    def _on_tool_cancel(self) -> None:
        self._deactivate_tools()
        self.show_status("Herramienta de recorte cancelada.")

    # ---------- herramienta esfera (primitiva) ---------- #

    def _on_esfera_movida(self, centro, radio: float) -> None:
        """El usuario soltó la esfera: se AJUSTA a los datos y se previsualiza.

        El ajuste solo ocurre aquí (al soltar). Mover los sliders reaplica la
        máscara sobre la esfera ya ajustada, sin re-ajustar: es más rápido y
        evita que la primitiva 'salte' mientras el usuario afina.
        """
        from app.modules.primitivas import ajustar_esfera

        if self._layer_stack is None:
            return
        pts = np.asarray(self._layer_stack.active.pcd.points)
        try:
            self._esfera = ajustar_esfera(pts, centro, radio,
                                          eps=self._esfera_tol)
        except RuntimeError as e:
            self._esfera = None
            self._ensure_crop_dock().set_esfera_has_selection(False)
            self.show_status(str(e))
            return
        self.show_status(f"Esfera ajustada: radio {self._esfera.radio:.3f} m.")
        self._repintar_esfera()

    def _on_esfera_params(self, tol: float, phi_min: float, phi_max: float) -> None:
        self._esfera_tol = tol
        self._esfera_phi = (phi_min, phi_max)
        self._repintar_esfera()

    def _repintar_esfera(self) -> None:
        """Reaplica la máscara con los parámetros actuales y actualiza el preview."""
        from app.modules.primitivas import mascara_esfera

        if self._esfera is None or self._layer_stack is None:
            return
        pts = np.asarray(self._layer_stack.active.pcd.points)
        phi_min, phi_max = self._esfera_phi
        self._esfera_mask = mascara_esfera(
            pts, self._esfera, self._esfera_tol, phi_min, phi_max
        )
        n = int(self._esfera_mask.sum())
        self._ensure_crop_dock().set_esfera_has_selection(n > 0)
        with self.viewer.camara_fija():
            self.viewer.preview_split(
                self._layer_stack.active.pcd, self._esfera_mask,
                f"layer_{self._layer_stack.active_index}",
            )
        self.show_status(
            f"Esfera r={self._esfera.radio:.3f} m · tol {self._esfera_tol:.2f} m · "
            f"φ [{phi_min:.0f}, {phi_max:.0f}] → {n:,} puntos capturados."
        )

    def _on_esfera_capturar(self) -> None:
        if self._esfera_mask is None or not self._esfera_mask.any():
            self.show_status("No hay puntos capturados.")
            return
        self._apply_split(self._esfera_mask)
        self._esfera = None
        self._esfera_mask = None
        self._ensure_crop_dock().set_esfera_has_selection(False)
        # La capa activa cambió: se reinicia la esfera sobre lo que queda.
        if self._active_tool == "esfera" and self._layer_stack is not None:
            self.viewer.start_sphere_widget(
                self._layer_stack.active.pcd, callback=self._on_esfera_movida
            )

    # ---------- herramienta plano (primitiva) ---------- #

    def _on_plano_movido(self, punto, normal) -> None:
        """El usuario soltó el plano: se ajusta a los datos y se previsualiza."""
        from app.modules.primitivas import ajustar_plano

        if self._layer_stack is None:
            return
        self._plano_pos = (np.asarray(punto, float), np.asarray(normal, float))
        pts = np.asarray(self._layer_stack.active.pcd.points)
        try:
            self._plano_base = ajustar_plano(pts, punto, normal,
                                             radio=self._plano_radio,
                                             eps=self._plano_tol)
        except RuntimeError as e:
            self._plano = self._plano_base = None
            self._ensure_crop_dock().set_plano_has_selection(False)
            self.show_status(str(e))
            return
        # el desplazamiento se mide desde el ajuste nuevo, no desde el anterior
        self._plano_offset = 0.0
        self._ensure_crop_dock().reset_plano_offset()
        self._rehacer_plano()
        self.show_status("Plano ajustado a la superficie.")
        self._repintar_plano()

    def _on_plano_params(self, tol: float, radio: float, offset: float) -> None:
        self._plano_tol = tol
        self._plano_radio = radio
        self._plano_offset = offset
        # El radio decide QUÉ PUNTOS entran al ajuste, no solo cuáles se
        # capturan; pero re-ajustar cuesta ~300 ms, así que solo se hace al
        # soltar el slider (señal plano_reajustar) o si no hay plano todavía.
        if self._plano_base is None and self._plano_pos is not None:
            self._reajustar_plano()
        self._rehacer_plano()
        self._repintar_plano()

    def _on_plano_reajustar(self) -> None:
        """El usuario soltó el slider de radio: recién ahí se re-ajusta."""
        self._reajustar_plano()
        self._rehacer_plano()
        self._repintar_plano()

    def _reajustar_plano(self) -> None:
        """Vuelve a ajustar con la última posición del widget y el radio actual."""
        from app.modules.primitivas import ajustar_plano

        if self._plano_pos is None or self._layer_stack is None:
            return
        punto, normal = self._plano_pos
        pts = np.asarray(self._layer_stack.active.pcd.points)
        try:
            self._plano_base = ajustar_plano(pts, punto, normal,
                                             radio=self._plano_radio,
                                             eps=self._plano_tol)
        except RuntimeError as e:
            self._plano_base = None
            self.show_status(str(e))

    def _rehacer_plano(self) -> None:
        """Reconstruye el plano con el radio y el desplazamiento actuales.

        El radio y el offset son parte de la PRIMITIVA (definen dónde está y
        hasta dónde llega), no de la máscara; por eso hay que rehacerla en vez
        de solo recalcular qué puntos caen dentro. Se parte siempre del plano
        ajustado (`_plano_base`), para que mover el slider de ida y vuelta
        devuelva exactamente al mismo sitio en vez de ir acumulando error.
        """
        from app.modules.primitivas import Plano

        if self._plano_base is None:
            return
        base = self._plano_base
        self._plano = Plano(
            normal=base.normal,
            d=base.d - self._plano_offset,          # desplazar por la normal
            centro=base.centro + self._plano_offset * base.normal,
            radio=self._plano_radio,
        )

    def _repintar_plano(self) -> None:
        from app.modules.primitivas import mascara_plano

        if self._plano is None or self._layer_stack is None:
            return
        pts = np.asarray(self._layer_stack.active.pcd.points)
        self._plano_mask = mascara_plano(pts, self._plano, self._plano_tol)
        n = int(self._plano_mask.sum())
        self._ensure_crop_dock().set_plano_has_selection(n > 0)
        with self.viewer.camara_fija():
            self.viewer.mostrar_disco_plano(
                self._plano.centro, self._plano.normal, self._plano.radio)
            self.viewer.preview_split(
                self._layer_stack.active.pcd, self._plano_mask,
                f"layer_{self._layer_stack.active_index}",
            )
        self.show_status(
            f"Plano · tol {self._plano_tol:.2f} m · radio {self._plano_radio:.1f} m · "
            f"desp {self._plano_offset:+.2f} m → {n:,} puntos capturados."
        )

    def _on_plano_capturar(self) -> None:
        if self._plano_mask is None or not self._plano_mask.any():
            self.show_status("No hay puntos capturados.")
            return
        self._apply_split(self._plano_mask)
        self._plano = self._plano_base = self._plano_mask = None
        self.viewer.ocultar_disco_plano()
        self.viewer.clear_preview()
        self._ensure_crop_dock().set_plano_has_selection(False)
        if self._active_tool == "plano" and self._layer_stack is not None:
            # sin ajuste inicial: repintar aquí haría parecer que la entidad
            # recién capturada quedó a medias (parte verde, parte roja)
            self._plano_pos = self.viewer.start_plane_widget(
                self._layer_stack.active.pcd, callback=self._on_plano_movido,
                ajustar_inicial=False,
            )
            self.show_status(
                f"Entidad capturada en '{self._layer_stack.activas[0].name}'. "
                "Coloca el plano de nuevo para la siguiente."
            )

    # ---------- herramienta cilindro (primitiva) ---------- #

    def _on_cilindro_movido(self, p1, normal_o_p2) -> None:
        """El usuario soltó la línea del eje: se ajusta el radio a los datos.

        El tramo de eje se toma de los extremos de la línea que puso el usuario:
        es más directo que un slider aparte, y ya expresa dónde empieza y
        termina el elemento.
        """
        from app.modules.primitivas import ajustar_cilindro

        if self._layer_stack is None:
            return
        p1, p2 = np.asarray(p1, float), np.asarray(normal_o_p2, float)
        eje = p2 - p1
        largo = float(np.linalg.norm(eje))
        if largo < 1e-6:
            self.show_status("La línea del eje es demasiado corta.")
            return
        eje = eje / largo

        pts = np.asarray(self._layer_stack.active.pcd.points)
        r, h, _ = self._radio_aprox(pts, p1, eje, largo)
        try:
            self._cilindro = ajustar_cilindro(pts, p1, eje, radio_aprox=r,
                                              h_min=0.0, h_max=largo,
                                              eps=self._cilindro_tol)
        except RuntimeError as e:
            self._cilindro = None
            self._ensure_crop_dock().set_cilindro_has_selection(False)
            self.show_status(str(e))
            return
        self._cilindro_h = (0.0, largo)
        self.show_status(f"Cilindro ajustado: radio {self._cilindro.radio:.3f} m.")
        self._repintar_cilindro()

    @staticmethod
    def _radio_aprox(pts, punto, eje, largo) -> tuple[float, np.ndarray, np.ndarray]:
        """Radio inicial para el ajuste: la mediana de la distancia al eje de los
        puntos que caen dentro del tramo. Arranca cerca de la superficie real en
        vez de un valor arbitrario que RANSAC tendría que salvar."""
        rel = pts - punto
        h = rel @ eje
        radial = np.linalg.norm(rel - np.outer(h, eje), axis=1)
        en_tramo = (h >= 0) & (h <= largo)
        if not en_tramo.any():
            return 1.0, radial, h
        return float(np.median(radial[en_tramo])), radial, h

    def _on_cilindro_params(self, tol: float, th_min: float, th_max: float) -> None:
        self._cilindro_tol = tol
        self._cilindro_theta = (th_min, th_max)
        self._repintar_cilindro()

    def _repintar_cilindro(self) -> None:
        from app.modules.primitivas import mascara_cilindro

        if self._cilindro is None or self._layer_stack is None:
            return
        pts = np.asarray(self._layer_stack.active.pcd.points)
        h_min, h_max = self._cilindro_h
        th_min, th_max = self._cilindro_theta
        self._cilindro_mask = mascara_cilindro(
            pts, self._cilindro, self._cilindro_tol, h_min, h_max, th_min, th_max
        )
        n = int(self._cilindro_mask.sum())
        self._ensure_crop_dock().set_cilindro_has_selection(n > 0)
        with self.viewer.camara_fija():
            self.viewer.mostrar_fantasma_cilindro(
                self._cilindro.punto, self._cilindro.eje, self._cilindro.radio,
                h_min, h_max)
            self.viewer.preview_split(
                self._layer_stack.active.pcd, self._cilindro_mask,
                f"layer_{self._layer_stack.active_index}",
            )
        self.show_status(
            f"Cilindro r={self._cilindro.radio:.3f} m · tol {self._cilindro_tol:.2f} m · "
            f"θ [{th_min:.0f}, {th_max:.0f}] → {n:,} puntos capturados."
        )

    def _on_cilindro_capturar(self) -> None:
        if self._cilindro_mask is None or not self._cilindro_mask.any():
            self.show_status("No hay puntos capturados.")
            return
        self._apply_split(self._cilindro_mask)
        self._cilindro = self._cilindro_mask = None
        self.viewer.ocultar_fantasma_cilindro()
        self.viewer.clear_preview()
        self._ensure_crop_dock().set_cilindro_has_selection(False)
        if self._active_tool == "cilindro" and self._layer_stack is not None:
            self.viewer.start_line_widget(
                self._layer_stack.active.pcd, callback=self._on_cilindro_movido
            )

    # ---------- herramienta cono (primitiva) ---------- #

    def _on_cono_movido(self, p1, p2) -> None:
        """El usuario soltó la línea del eje: se ajusta la apertura a los datos."""
        from app.modules.primitivas import ajustar_cono

        if self._layer_stack is None:
            return
        p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
        eje = p2 - p1
        largo = float(np.linalg.norm(eje))
        if largo < 1e-6:
            self.show_status("La línea del eje es demasiado corta.")
            return
        eje = eje / largo

        pts = np.asarray(self._layer_stack.active.pcd.points)
        r, _, _ = self._radio_aprox(pts, p1, eje, largo)
        try:
            self._cono_base = ajustar_cono(pts, p1, eje, radio_aprox=r,
                                           h_min=0.0, h_max=largo,
                                           eps=self._cono_tol)
        except RuntimeError as e:
            self._cono = self._cono_base = None
            self._ensure_crop_dock().set_cono_has_selection(False)
            self.show_status(str(e))
            return
        self._cono_h = (0.0, largo)
        self._volver_a_la_apertura_ajustada()
        self._repintar_cono()

    def _volver_a_la_apertura_ajustada(self) -> None:
        """El slider pasa a mostrar la apertura que encontró RANSAC, que es
        desde donde el usuario va a corregir."""
        if self._cono_base is None:
            return
        signo = float(np.degrees(np.arctan(self._cono_base.pendiente)))
        self._cono_apertura = signo
        self._cono = self._cono_base
        self._ensure_crop_dock().set_cono_apertura(signo)

    def _on_cono_reajustar(self) -> None:
        self._volver_a_la_apertura_ajustada()
        self._repintar_cono()

    def _on_cono_params(self, tol: float, th_min: float, th_max: float,
                        apertura: float) -> None:
        from app.modules.primitivas import con_apertura
        self._cono_tol = tol
        self._cono_theta = (th_min, th_max)
        if self._cono_base is not None and apertura != self._cono_apertura:
            self._cono_apertura = apertura
            self._cono = con_apertura(self._cono_base, apertura, *self._cono_h)
        self._repintar_cono()

    def _repintar_cono(self) -> None:
        from app.modules.primitivas import mascara_cono

        if self._cono is None or self._layer_stack is None:
            return
        pts = np.asarray(self._layer_stack.active.pcd.points)
        h_min, h_max = self._cono_h
        th_min, th_max = self._cono_theta
        self._cono_mask = mascara_cono(pts, self._cono, self._cono_tol,
                                       h_min, h_max, th_min, th_max)
        n = int(self._cono_mask.sum())
        dock = self._ensure_crop_dock()
        dock.set_cono_has_selection(n > 0)
        # un cono de pendiente ~0 es en realidad un cilindro: conviene decirlo en
        # vez de dejar que el usuario lo descubra por su cuenta
        dock.set_cono_info(
            f"Semiángulo {self._cono.semiangulo:.1f}° · "
            + ("sin apertura: es un cilindro" if self._cono.vertice() is None
               else f"radio {float(self._cono.radio_en(h_min)):.2f} → "
                    f"{float(self._cono.radio_en(h_max)):.2f} m")
        )
        with self.viewer.camara_fija():
            self.viewer.mostrar_fantasma_cono(
                self._cono.punto, self._cono.eje,
                float(self._cono.radio_en(h_min)),
                float(self._cono.radio_en(h_max)), h_min, h_max)
            self.viewer.preview_split(
                self._layer_stack.active.pcd, self._cono_mask,
                f"layer_{self._layer_stack.active_index}",
            )
        self.show_status(
            f"Cono · semiángulo {self._cono.semiangulo:.1f}° · "
            f"tol {self._cono_tol:.2f} m · θ [{th_min:.0f}, {th_max:.0f}] → "
            f"{n:,} puntos capturados."
        )

    def _on_cono_capturar(self) -> None:
        if self._cono_mask is None or not self._cono_mask.any():
            self.show_status("No hay puntos capturados.")
            return
        self._apply_split(self._cono_mask)
        self._cono = self._cono_base = self._cono_mask = None
        self.viewer.ocultar_fantasma_cono()
        self.viewer.ocultar_malla()
        self.viewer.clear_preview()
        self._ensure_crop_dock().set_cono_has_selection(False)
        if self._active_tool == "cono" and self._layer_stack is not None:
            self.viewer.start_line_widget(
                self._layer_stack.active.pcd, callback=self._on_cono_movido
            )

    # ---------- herramienta DBSCAN ---------- #

    def _mostrar_referencia_dbscan(self) -> None:
        """Muestra el espaciado de la nube: es la referencia para elegir eps.

        Sin este dato el usuario ajusta a ciegas, y eps es el parámetro que
        decide si todo queda en un solo cluster o si todo queda como ruido.
        """
        from app.modules.clusters import MAX_PUNTOS_DBSCAN, espaciado_efectivo

        if self._layer_stack is None:
            return
        pts = np.asarray(self._layer_stack.active.pcd.points)
        if len(pts) < 2:
            return
        # el espaciado se mide sobre la submuestra que DBSCAN va a ver, no
        # sobre la nube completa: si no, el eps sugerido queda muy por debajo
        e = espaciado_efectivo(pts)
        nota = ("" if len(pts) <= MAX_PUNTOS_DBSCAN
                else f" (se agrupan {MAX_PUNTOS_DBSCAN:,} y se propaga al resto)")
        dock = self._ensure_crop_dock()
        dock.set_dbscan_referencia(
            f"{len(pts):,} puntos{nota} · espaciado {e:.3f} m. "
            f"Un eps entre {2*e:.2f} y {5*e:.2f} m suele funcionar."
        )
        self._db_espaciado = e
        from app.modules.clusters import EPS_POR_ESPACIADO
        dock.set_dbscan_eps(EPS_POR_ESPACIADO * e)

    def _on_dbscan_eps(self, eps: float) -> None:
        """Con cada eps cambia cuántos vecinos tiene un punto, y ese número es
        el techo de min_points: por encima, ningún punto llega a ser núcleo y
        todo queda como ruido."""
        from app.modules.clusters import MAX_PUNTOS_DBSCAN, vecinos_tipicos

        if self._layer_stack is None or getattr(self, "_db_espaciado", 0) <= 0:
            return
        pts = np.asarray(self._layer_stack.active.pcd.points)
        if len(pts) > MAX_PUNTOS_DBSCAN:
            idx = np.random.default_rng(0).choice(len(pts), MAX_PUNTOS_DBSCAN,
                                                  replace=False)
            pts = pts[idx]
        from app.modules.clusters import MIN_POINTS_POR_VECINOS
        v = vecinos_tipicos(pts, eps)
        sugerido = max(3, int(round(v * MIN_POINTS_POR_VECINOS)))
        self._ensure_crop_dock().set_dbscan_referencia(
            f"{len(pts):,} puntos agrupables · espaciado {self._db_espaciado:.3f} m.\n"
            f"Con este eps un punto tiene ~{v} vecinos → min_points {sugerido} "
            f"(la proporción de la corrida buena del interior).\n"
            f"OJO: DBSCAN separa lo que está DESCONECTADO. Sobre la estructura "
            f"completa dará un solo cluster; úsalo después de quitar la cúpula, "
            f"el tambor y el suelo con las primitivas."
        )

    def _on_dbscan_agrupar(self, eps: float, min_points: int) -> None:
        if self._layer_stack is None:
            return
        pts = np.asarray(self._layer_stack.active.pcd.points)
        self.show_status(f"Agrupando {len(pts):,} puntos (eps={eps:.2f}, "
                         f"min_points={min_points})...")
        self._set_loading(True)

        from app.core.workers import DbscanWorker
        worker = DbscanWorker(pts, eps, min_points, parent=self)
        worker.finished.connect(self._on_dbscan_listo)
        worker.error.connect(self._on_load_error)
        self._current_worker = worker
        worker.start()

    def _on_dbscan_listo(self, etiquetas) -> None:
        self._set_loading(False)
        self._db_etiquetas = etiquetas
        self._db_seleccion = []
        self._refrescar_clusters()

    def _on_dbscan_umbral(self, minimo: int) -> None:
        self._db_umbral = int(minimo)
        # el umbral se aplica sobre las etiquetas ya calculadas: no se repite
        # DBSCAN, que es lo caro
        self._refrescar_clusters()

    def _refrescar_clusters(self) -> None:
        from app.modules.clusters import info_clusters, marcar_pequenos_como_ruido

        if self._db_etiquetas is None or self._layer_stack is None:
            return
        et = self._db_etiquetas
        if self._db_umbral > 0:
            et = marcar_pequenos_como_ruido(et, self._db_umbral)
        self._db_etiquetas_filtradas = et

        pts = np.asarray(self._layer_stack.active.pcd.points)
        infos = info_clusters(pts, et)
        dock = self._ensure_crop_dock()
        dock.refresh_clusters(infos)
        self._db_seleccion = []
        self._pintar_clusters()
        n_ruido = int((et == -1).sum())
        self.show_status(
            f"{len(infos)} clusters · {n_ruido:,} puntos como ruido. "
            "Marca los que quieras y usa Capturar o Eliminar."
        )

    def _pintar_clusters(self) -> None:
        if self._db_etiquetas_filtradas is None or self._layer_stack is None:
            return
        with self.viewer.camara_fija():
            self.viewer.mostrar_clusters(
                self._layer_stack.active.pcd, self._db_etiquetas_filtradas,
                self._db_seleccion, f"layer_{self._layer_stack.active_index}",
            )

    def _on_dbscan_seleccion(self, ids) -> None:
        self._db_seleccion = [int(i) for i in ids]
        self._pintar_clusters()
        if self._db_etiquetas_filtradas is not None:
            from app.modules.clusters import mascara_de
            n = int(mascara_de(self._db_etiquetas_filtradas, self._db_seleccion).sum())
            self.show_status(f"{len(self._db_seleccion)} clusters marcados · "
                             f"{n:,} puntos.")

    def _mascara_seleccion_dbscan(self):
        from app.modules.clusters import mascara_de
        if self._db_etiquetas_filtradas is None or not self._db_seleccion:
            return None
        return mascara_de(self._db_etiquetas_filtradas, self._db_seleccion)

    def _on_dbscan_capturar(self) -> None:
        """Cada cluster marcado pasa a ser su propia capa.

        Por separado y no fusionados: un cluster es una unidad que el usuario
        puede juzgar, y siempre puede unir varias capas después. Al revés no:
        una capa fusionada ya no se puede volver a separar.
        """
        from app.core.layers import CloudLayer, _subset
        from app.modules.clusters import mascara_de

        if self._layer_stack is None or not self._db_seleccion:
            return
        stack = self._layer_stack
        et = self._db_etiquetas_filtradas
        usados = mascara_de(et, self._db_seleccion)
        if not usados.any():
            self.show_status("Los clusters marcados no tienen puntos.")
            return

        try:
            nuevas = stack.extraer_de_activas(
                [(f"Cluster {cid}", et == cid) for cid in self._db_seleccion])
        except ValueError as e:
            self.show_status(str(e))
            return

        self._db_etiquetas = self._db_etiquetas_filtradas = None
        self._db_seleccion = []
        self.viewer.ocultar_clusters()
        self._redibujar_capas(stack)
        dock = self._ensure_crop_dock()
        self._ensure_layer_dock().refresh_layers(stack)
        dock.refresh_clusters([])
        self._mostrar_referencia_dbscan()
        self.show_status(
            f"{len(nuevas)} clusters capturados como capas. Renombralos con "
            "doble clic, o marca varios y usa 'Unir seleccionadas'."
        )

    def _on_dbscan_eliminar(self) -> None:
        """Quita de la capa los clusters marcados. La otra mitad de la feature:
        limpiar ruido y objetos que no son estructura."""
        from app.core.layers import _subset

        mask = self._mascara_seleccion_dbscan()
        if mask is None or not mask.any() or self._layer_stack is None:
            return
        if mask.all():
            self.show_status("Eso eliminaria la capa entera.")
            return
        stack = self._layer_stack
        try:
            n = stack.eliminar_de_activas(mask)
        except ValueError as e:
            self.show_status(str(e))
            return

        self._db_etiquetas = self._db_etiquetas_filtradas = None
        self._db_seleccion = []
        self.viewer.ocultar_clusters()
        self._redibujar_capas(stack)
        dock = self._ensure_crop_dock()
        self._ensure_layer_dock().refresh_layers(stack)
        dock.refresh_clusters([])
        self._mostrar_referencia_dbscan()
        self.show_status(f"{n:,} puntos eliminados. Vuelve a agrupar si quieres "
                         "seguir limpiando.")

    def _refrescar_visibilidad(self) -> None:
        """Redibuja tras un cambio de visibilidad en bloque."""
        stack = self._layer_stack
        self._redibujar_capas(stack)
        self._ensure_layer_dock().refresh_layers(stack)
        n = sum(1 for c in stack.layers if c.visible)
        self.show_status(f"{n} de {len(stack.layers)} capas visibles. "
                         f"{self._resumen_en_uso()}")

    def _on_layers_visibility(self, indices, visible: bool) -> None:
        if self._layer_stack is None:
            return
        try:
            self._layer_stack.set_visibles(indices, visible)
        except IndexError as e:
            self.show_status(str(e))
            return
        self._refrescar_visibilidad()

    def _on_layers_isolate(self, indices) -> None:
        if self._layer_stack is None:
            return
        try:
            self._layer_stack.mostrar_solo(indices)
        except (ValueError, IndexError) as e:
            self.show_status(str(e))
            return
        self._refrescar_visibilidad()

    def _on_layers_invert(self) -> None:
        if self._layer_stack is None:
            return
        self._layer_stack.invertir_visibilidad()
        self._refrescar_visibilidad()

    def _on_layers_merge(self, indices) -> None:
        """Une varias capas en una entidad."""
        if self._layer_stack is None or len(indices) < 2:
            return
        sugerido = self._layer_stack.layers[indices[0]].name
        nombre, ok = QInputDialog.getText(
            self, "Unir capas",
            f"Se unirán {len(indices)} capas en una.\n\nNombre de la entidad:",
            text=sugerido)
        if not ok:
            return
        try:
            capa = self._layer_stack.unir(indices, nombre)
        except (ValueError, IndexError) as e:
            QMessageBox.warning(self, "Unir capas", str(e))
            return
        self._redibujar_capas(self._layer_stack)
        self._ensure_layer_dock().refresh_layers(self._layer_stack)
        self.show_status(f"{len(indices)} capas unidas en "
                         f"'{capa.name}' ({len(capa.pcd.points):,} pts).")

    # ---------- reparación por simetría ---------- #

    def _zona_simetria(self):
        """Caja que acota dónde aplica la simetría, o None."""
        if not self._si_zona or self._crop_min is None:
            return None
        return (self._crop_min, self._crop_max)

    def _on_simetria_plano_movido(self, punto, normal) -> None:
        from app.modules.simetria import PlanoSimetria
        self._si_plano = PlanoSimetria(normal=normal, punto=punto)
        self._evaluar_simetria()

    def _on_simetria_params(self, tolerancia: float, zona: bool) -> None:
        self._si_tol = tolerancia
        self._si_zona = zona
        if zona and self._crop_min is None and self._layer_stack is not None:
            # la caja del recorte es la que acota: si no hay ninguna, se ofrece
            self.viewer.start_crop_widget(
                self._layer_stack.active.pcd,
                callback=self._on_crop_bounds_changed)
            self.show_status("Ajusta la caja para acotar dónde aplica la simetría.")
        self._evaluar_simetria()

    def _evaluar_simetria(self) -> None:
        """Recalcula concordancia y relleno con el plano actual.

        No refina: refinar es la operación cara y va en su propio botón. Esto
        solo mide, para que el usuario vea de inmediato si va por buen camino.
        """
        from app.modules.simetria import concordancia, puntos_a_rellenar

        if self._si_plano is None or self._layer_stack is None:
            return
        pts = np.asarray(self._layer_stack.active.pcd.points)
        zona = self._zona_simetria()
        conc = concordancia(pts, self._si_plano, self._si_tol, zona)
        self._si_relleno = puntos_a_rellenar(pts, self._si_plano, self._si_tol, zona)
        n = len(self._si_relleno)
        self._ensure_crop_dock().set_simetria_concordancia(conc, n)
        with self.viewer.camara_fija():
            self.viewer.mostrar_relleno(self._si_relleno)
        self.show_status(
            f"Concordancia {100*conc:.1f}% · el relleno agregaría {n:,} puntos. "
            + ("Concordancia baja: revisa el plano antes de aplicar."
               if conc < 0.5 else "Pulsa 'Refinar plano' para afinarlo.")
        )

    def _on_simetria_refinar(self) -> None:
        if self._si_plano is None or self._layer_stack is None:
            self.show_status("Coloca primero el plano de simetría.")
            return
        pts = np.asarray(self._layer_stack.active.pcd.points)
        self.show_status(f"Refinando el plano sobre {len(pts):,} puntos...")
        self._set_loading(True)

        from app.core.workers import SimetriaWorker
        worker = SimetriaWorker(pts, self._si_plano, self._si_tol,
                                self._zona_simetria(), parent=self)
        worker.finished.connect(self._on_simetria_refinado)
        worker.error.connect(self._on_load_error)
        self._current_worker = worker
        worker.start()

    def _on_simetria_refinado(self, plano, conc: float) -> None:
        self._set_loading(False)
        self._si_plano = plano
        self._evaluar_simetria()

    def _on_simetria_aplicar(self) -> None:
        """El relleno va a una CAPA APARTE, no se mezcla con los datos medidos.

        Es material generado: mezclarlo con lo medido haría imposible saber
        después qué se midió y qué se inventó. Una vez revisado, el usuario lo
        une a la entidad con 'Unir seleccionadas'.
        """
        import open3d as o3d
        from app.core.layers import CloudLayer

        if self._si_relleno is None or len(self._si_relleno) == 0:
            self.show_status("No hay puntos de relleno que crear.")
            return
        stack = self._layer_stack
        nombre = f"{stack.activas[0].name} · relleno"
        capa = CloudLayer(
            name=nombre,
            pcd=o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(self._si_relleno)),
        )
        i = stack.active_index + 1
        stack.layers.insert(i, capa)
        stack.active_index = i

        n = len(self._si_relleno)
        self._si_plano = self._si_relleno = None
        self.viewer.ocultar_relleno()
        self._redibujar_capas(stack)
        dock = self._ensure_crop_dock()
        self._ensure_layer_dock().refresh_layers(stack)
        dock.set_simetria_concordancia(None)
        self.show_status(
            f"Capa '{nombre}' creada con {n:,} puntos generados. Revísala y "
            "únela a la entidad si estás conforme."
        )

    # ---------- herramienta lazo ---------- #

    def _on_lasso_started(self, set_op: str) -> None:
        if self._layer_stack is None:
            return
        self._lasso_set_op = set_op
        if self._lasso_mask is None:
            self._lasso_mask = np.zeros(
                len(self._layer_stack.active.pcd.points), dtype=bool
            )
        self._ensure_crop_dock().set_lasso_active(True)
        for act in self._acts_vista:
            act.setEnabled(False)
        self.viewer.start_lasso(self._on_lasso_polygon_closed)
        self.show_status(
            "Haz clic para agregar vertices. Cierra el lazo con clic derecho, "
            "doble clic, o clicando sobre el primer vertice."
        )

    def _on_lasso_polygon_closed(self, verts: list[tuple[int, int]]) -> None:
        if self._layer_stack is None or self._lasso_mask is None:
            return

        from app.modules.processing import apply_lasso

        for act in self._acts_vista:
            act.setEnabled(True)

        if len(verts) < 3:
            self._ensure_crop_dock().set_lasso_active(False)
            self.show_status("Lazo descartado: se necesitan al menos 3 vertices.")
            return

        activa = self._layer_stack.active
        screen_pts, valid = self.viewer.project_cloud_to_screen(activa.pcd)

        # Diagnóstico visible: cuántos centros de puntos caen dentro del polígono
        from matplotlib.path import Path as _MplPath
        n_dentro = int((_MplPath(verts).contains_points(screen_pts) & valid).sum())
        print(f"[lazo] vertices={len(verts)}, puntos dentro del poligono={n_dentro:,}, "
              f"op={self._lasso_set_op}, capa='{activa.name}' ({len(screen_pts):,} pts)")

        # Sin seleccion previa, Quitar/Intersectar parten de la capa completa:
        # "quitar estos puntos" significa "todo menos esto".
        mask_base = self._lasso_mask
        if not mask_base.any() and self._lasso_set_op in ("difference", "intersection"):
            mask_base = np.ones(len(mask_base), dtype=bool)

        dock = self._ensure_crop_dock()
        try:
            new_mask = apply_lasso(
                screen_pts, valid, verts, self._lasso_set_op, mask_base
            )
        except ValueError as e:
            self.show_status(str(e))
            dock.set_lasso_active(False)
            return

        if not new_mask.any():
            self.show_status(
                f"Lazo: {n_dentro:,} pts dentro del poligono; la operacion dejo la "
                "seleccion vacia. Seleccion sin cambios."
            )
        else:
            self._lasso_mask = new_mask
            # Guardar la operación para reconstruir el recorte sobre la nube fina
            self._lasso_ops.append((verts, self._lasso_set_op))
            self.show_status(
                f"Lazo: {n_dentro:,} pts dentro del poligono → "
                f"{int(new_mask.sum()):,} en verde. Puedes mover la camara para "
                "revisar; 'Aplicar recorte' confirma."
            )

        dock.set_lasso_active(False)
        dock.set_lasso_has_selection(bool(self._lasso_mask.any()))
        self.viewer.preview_split(
            activa.pcd, self._lasso_mask,
            f"layer_{self._layer_stack.active_index}",
        )

    def _on_lasso_apply(self) -> None:
        if self._lasso_mask is None or not self._lasso_mask.any():
            self.show_status("No hay puntos seleccionados.")
            return
        self._apply_split(self._lasso_mask)

    def _on_lasso_cancel(self) -> None:
        for act in self._acts_vista:
            act.setEnabled(True)
        self.viewer.stop_lasso()
        self.viewer.clear_preview()
        self._lasso_mask = None
        dock = self._ensure_crop_dock()
        dock.set_lasso_active(False)
        dock.set_lasso_has_selection(False)
        self.show_status("Lazo cancelado.")

    # ---------- capas ---------- #

    def _on_layer_visibility(self, i: int, visible: bool) -> None:
        if self._layer_stack is None:
            return
        stack = self._layer_stack
        stack.set_visible(i, visible)
        self._redibujar_capas(stack)
        # No se repuebla la lista —eso perdería el desplazamiento y la selección—
        # pero el recuento sí debe seguir al estado, o queda mintiendo.
        n = sum(1 for c in stack.layers if c.visible)
        self.show_status(f"{n} de {len(stack.layers)} capas visibles. "
                         f"{self._resumen_en_uso()}")

    # --------------------------------------------------------------- malla

    def _mostrar_referencia_malla(self) -> None:
        """Propone el límite de perímetro a partir del espaciado de la entidad."""
        stack = self._layer_stack
        dock = self._ensure_crop_dock()
        if stack is None or not stack.active_indices:
            dock.set_malla_referencia(None)
            return
        pts = np.asarray(stack.active.pcd.points)
        if len(pts) < 2:
            dock.set_malla_referencia(None)
            return
        from app.modules.meshing import espaciado_medio
        dock.set_malla_referencia(espaciado_medio(pts), len(pts))

    def _on_malla_generar(self, perimetro: float, rellenar: bool,
                          perimetro_relleno: float) -> None:
        """Triangula las capas en uso, en un hilo aparte."""
        stack = self._layer_stack
        if stack is None or not stack.active_indices:
            self.show_status("Marca en 'Usar' la entidad que quieras mallar.")
            return
        pts = np.asarray(stack.active.pcd.points)
        if len(pts) < 4:
            self.show_status("Hacen falta al menos cuatro puntos para triangular.")
            return

        from app.core.workers import MallaWorker
        dock = self._ensure_crop_dock()
        dock.set_malla_ocupado(True)
        dock.set_malla_resultado("Triangulando…")
        self.show_status(f"Generando malla de {len(pts):,} puntos…")
        worker = MallaWorker(pts, perimetro, rellenar, perimetro_relleno,
                             parent=self)
        worker.finished.connect(self._on_malla_lista)
        worker.error.connect(self._on_malla_error)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        self._malla_worker = worker
        worker.start()

    def _on_malla_lista(self, malla, fidelidad, stats) -> None:
        self._malla = malla
        dock = self._ensure_crop_dock()
        dock.set_malla_ocupado(False)

        # La fidelidad se muestra siempre, no solo si es buena: es el dato que
        # permite decidir si la malla sirve, y esconderlo cuando sale mal es
        # justamente lo que haría inútil haberla medido.
        stack = self._layer_stack
        n_puntos = (sum(len(c.pcd.points) for c in stack.activas)
                    if stack is not None else 0)
        dock.set_malla_metricas(malla.n_caras, malla.n_vertices, n_puntos,
                                fidelidad, stats)
        self._on_malla_vista(dock.malla_vista())
        self.show_status(
            f"Malla generada: {malla.n_caras:,} triángulos, "
            f"{fidelidad.invencion * 100:.2f} % de superficie inventada. "
            "Revísala y expórtala si estás conforme.")

    def _on_malla_vista(self, modo: str) -> None:
        """Qué se dibuja tras generar: la malla, la nube o ambas.

        Existe porque Advancing Front INTERPOLA los puntos: la superficie pasa
        exactamente por cada uno, de modo que con la nube dibujada encima la
        malla queda tapada por completo y el resultado parece idéntico a no
        haber hecho nada.
        """
        malla = getattr(self, "_malla", None)
        if malla is None:
            return
        if modo == "nube":
            self.viewer.ocultar_malla()
            return
        # Semitransparente solo cuando conviven; sola va opaca y sombreada, que
        # es como se juzga si la superficie es correcta.
        self.viewer.mostrar_malla(malla.vertices, malla.caras,
                                  opacidad=0.55 if modo == "ambas" else 1.0)
        self.viewer.set_capas_visibles(modo == "ambas")

    def _on_malla_error(self, mensaje: str) -> None:
        self._malla = None
        dock = self._ensure_crop_dock()
        dock.set_malla_ocupado(False)
        dock.set_malla_resultado(f"No se pudo generar: {mensaje}")
        self.show_status(f"Malla: {mensaje}")

    def _on_malla_exportar(self) -> None:
        malla = getattr(self, "_malla", None)
        if malla is None:
            self.show_status("No hay ninguna malla generada que exportar.")
            return
        stack = self._layer_stack
        base = _slug(stack.activas[0].name) if stack and stack.activas else "malla"
        destino, _ = QFileDialog.getSaveFileName(
            self, "Guardar malla", f"{base}.ply", "Malla PLY (*.ply)")
        if not destino:
            return
        from app.modules.meshing import exportar
        try:
            ruta = exportar(malla, destino)
        except Exception as ex:
            self.show_status(f"No se pudo guardar: {ex}")
            return
        self.show_status(f"Malla guardada en {ruta} "
                         f"({malla.n_caras:,} triángulos).")

    def _resumen_en_uso(self) -> str:
        """Frase para la barra de estado, válida también sin capas en uso.

        El conjunto vacío es legítimo desde que 'Ninguna' actúa sobre 'Usar', y
        los mensajes de visibilidad se emiten con o sin capas en uso: dar por
        hecho que hay al menos una reventaba al apagar la última.
        """
        stack = self._layer_stack
        if stack is None or not stack.active_indices:
            return "Ninguna capa en uso."
        if len(stack.active_indices) == 1:
            return f"En uso: '{stack.activas[0].name}'."
        return f"{len(stack.active_indices)} capas en uso."

    def _redibujar_capas(self, stack=None) -> None:
        """Redibuja y pone al día el resumen del panel.

        Van juntos a propósito: el resumen se calculaba una sola vez al cargar,
        así que el panel seguía anunciando los puntos de la carga inicial aunque
        se apagaran todas las capas. Cualquier cambio del stack pasa por aquí.
        """
        stack = stack if stack is not None else self._layer_stack
        if stack is None:
            return
        self.viewer.show_layers(stack)
        self.info_panel.update_layer_summary(stack.layers, stack.activas)

    def _on_activate_visible(self) -> None:
        """'Usar' ← 'Ver': operar justo sobre lo que se está viendo.

        Es el caso más frecuente, y con dos columnas independientes costaría
        marcar una por una.
        """
        stack = self._layer_stack
        if stack is None:
            return
        visibles = [i for i, c in enumerate(stack.layers) if c.visible]
        if not visibles:
            self.show_status("No hay ninguna capa visible que activar.")
            return
        self._on_layers_activated(visibles)
        self._ensure_layer_dock().marcar_activas(visibles)

    def _on_layer_activated(self, i: int) -> None:
        self._on_layers_activated([i])

    def _on_layers_activated(self, filas: list[int]) -> None:
        """La selección de la lista ES el conjunto de capas activas.

        Puede ser más de una: si una entidad quedó repartida entre varios
        escaneos, las herramientas deben poder tratarlos juntos. Una selección
        vacía se ignora y se conserva la activa anterior, porque toda
        herramienta necesita al menos una capa de la que partir.
        """
        stack = self._layer_stack
        if stack is None:
            return
        filas = [int(f) for f in filas if 0 <= f < len(stack.layers)]
        if filas == stack.active_indices:
            return
        if not filas:
            # Vaciar el conjunto es un paso legítimo —limpiar para rearmar— pero
            # deja a la herramienta abierta sin material, así que se cierra en
            # vez de dejarla apuntando a nada.
            stack.set_activas([])
            self._ensure_layer_dock().marcar_activas([])
            if self._active_tool is not None:
                self._deactivate_tools(keep_checked=None)
                self.viewer.clear_preview()
                if self._crop_dock is not None:
                    self._crop_dock.hide()
            self._redibujar_capas(stack)
            self.show_status(
                "Ninguna capa en uso. Marca las que quieras en la columna "
                "'Usar' para volver a trabajar.")
            return
        stack.set_activas(filas)
        self._ensure_layer_dock().marcar_activas(filas)
        self._lasso_mask = None
        self.viewer.stop_lasso()
        self.viewer.clear_preview()
        dock = self._ensure_crop_dock()
        dock.set_lasso_active(False)
        dock.set_lasso_has_selection(False)
        self._redibujar_capas(stack)
        if self._active_tool == "caja":
            self.viewer.start_crop_widget(
                stack.active.pcd, callback=self._on_crop_bounds_changed
            )
        n = len(stack.active_indices)
        # El recuento se suma capa por capa a propósito: pedirlo a `stack.active`
        # materializaría la unión de los puntos, y con 35 capas eso es una copia
        # de 600 MB por cada cambio de selección. La unión se construye cuando
        # una herramienta la necesita de verdad, no al seleccionar.
        pts = sum(len(c.pcd.points) for c in stack.activas)
        self.show_status(
            f"Capa activa: '{stack.activas[0].name}' ({pts:,} pts)." if n == 1 else
            f"{n} capas activas ({pts:,} pts). Las herramientas operan sobre todas.")

    def _on_layer_removed(self, i: int) -> None:
        self._on_layers_removed([i])

    def _on_layers_removed(self, filas: list[int]) -> None:
        """Elimina TODAS las capas indicadas, que son las seleccionadas.

        Borrar solo la fila actual teniendo doce marcadas no corresponde a lo
        que el usuario ve, y es destructivo: hay que acertar a la primera.
        """
        stack = self._layer_stack
        if stack is None:
            return
        filas = sorted({int(f) for f in filas if 0 <= f < len(stack.layers)})
        if not filas:
            return
        if len(filas) >= len(stack.layers):
            self.show_status("No se pueden eliminar todas las capas: debe quedar una.")
            return

        capas = [stack.layers[i] for i in filas]
        pts = sum(len(c.pcd.points) for c in capas)
        if len(capas) == 1:
            detalle = f"la capa '{capas[0].name}' ({pts:,} puntos)"
        else:
            nombres = ", ".join(f"'{c.name}'" for c in capas[:4])
            if len(capas) > 4:
                nombres += f" y {len(capas) - 4} más"
            detalle = f"{len(capas)} capas ({pts:,} puntos):\n{nombres}"
        reply = QMessageBox.question(
            self,
            "Eliminar capa" if len(capas) == 1 else "Eliminar capas",
            f"¿Eliminar {detalle}?\n"
            "Podrás deshacerlo con 'Restaurar eliminada' mientras no elimines otra.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        # De mayor a menor: borrar de atrás hacia adelante deja intactos los
        # índices que todavía no se han usado.
        try:
            for i in reversed(filas):
                stack.remove(i)
        except ValueError as e:
            self.show_status(str(e))
            return
        self._removed_layer_backup = list(zip(filas, capas))
        self._lasso_mask = None
        self.viewer.clear_preview()
        self._redibujar_capas(stack)
        dock = self._ensure_layer_dock()
        dock.refresh_layers(stack)
        dock.set_restore_enabled(True)
        self.show_status(
            f"Capa '{capas[0].name}' eliminada (puedes restaurarla)."
            if len(capas) == 1 else
            f"{len(capas)} capas eliminadas, {pts:,} puntos "
            "(puedes restaurarlas todas).")

    def _on_layer_restore(self) -> None:
        stack = self._layer_stack
        backup = getattr(self, "_removed_layer_backup", None)
        if stack is None or backup is None:
            return
        # De menor a mayor, al revés que al eliminar: cada inserción desplaza a
        # las posteriores, así que hay que ir en orden ascendente.
        for i, capa in sorted(backup, key=lambda x: x[0]):
            stack.insert_layer(i, capa)
        self._removed_layer_backup = None
        self._redibujar_capas(stack)
        dock = self._ensure_layer_dock()
        dock.refresh_layers(stack)
        dock.set_restore_enabled(False)
        self.show_status(
            f"Capa '{backup[0][1].name}' restaurada." if len(backup) == 1
            else f"{len(backup)} capas restauradas.")

    def _on_export_visible(self) -> None:
        """Une las capas visibles y las exporta a un único .ply."""
        from app.core.io import export_point_cloud
        if self._layer_stack is None:
            return
        try:
            merged = self._layer_stack.merge_visible_fine()
        except ValueError as e:
            self.show_status(str(e))
            return
        last_dir = self.settings.value("io/last_dir", os.path.expanduser("~"))
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar capas visibles", last_dir, "PLY (*.ply)"
        )
        if not path:
            return
        if not path.lower().endswith(".ply"):
            path += ".ply"
        try:
            export_point_cloud(merged, path)
            self.settings.setValue("io/last_dir", os.path.dirname(path))
            self.show_status(
                f"Exportadas {len(merged.points):,} pts de capas visibles "
                f"(resolución fina): {path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error al exportar", str(e))

    # ------------------------------------------------------------------ #
    # Persistencia                                                         #
    # ------------------------------------------------------------------ #

    def closeEvent(self, event):
        self.settings.setValue("window/geometry", self.saveGeometry())
        super().closeEvent(event)
