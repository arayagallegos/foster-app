"""
main_window.py — Ventana principal de Foster App.

Layout:
  ┌──────────────────────────────────────────────────┐
  │  Toolbar (cargar LiDAR, cargar foto, limpiar...) │
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
    QSplitter, QToolBar, QStatusBar,
    QProgressBar, QLabel, QFileDialog,
    QMessageBox, QApplication,
    QMenu, QToolButton,
)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence

from app.core.layers import LayerStack
from app.core.project import Project
from app.core.workers import LoadWorker
from app.gui.viewer import Viewer3D
from app.gui.panels.crop_dock import CropDock
from app.gui.panels.info_panel import InfoPanel


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
        self._layer_stack: LayerStack | None = None
        self._active_tool: str | None = None   # None | "caja" | "lazo"
        self._crop_min: np.ndarray | None = None
        self._crop_max: np.ndarray | None = None
        self._lasso_mask: np.ndarray | None = None
        self._lasso_ops: list = []   # [(verts, set_op)] para reconstruir el recorte fino
        self._lasso_set_op: str = "union"

        self._setup_window()
        self._setup_central()
        self._setup_toolbar()
        self._setup_statusbar()
        self._apply_stylesheet()

        self.setWindowTitle(f"{self.APP_NAME} v{self.VERSION} — Sin título")
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

        # --- Cargar LiDAR ---
        self._act_load_lidar = QAction("📡  Cargar LiDAR", self)
        self._act_load_lidar.setShortcut(QKeySequence("Ctrl+L"))
        self._act_load_lidar.setToolTip(
            "Cargar nube de puntos LiDAR (.e57, .las, .laz, .ply)\n"
            "Atajo: Ctrl+L"
        )
        self._act_load_lidar.triggered.connect(self._on_load_lidar)
        tb.addAction(self._act_load_lidar)

        # --- Registrar scans e57 ---
        self._act_register = QAction("🔗  Registrar scans", self)
        self._act_register.setShortcut(QKeySequence("Ctrl+R"))
        self._act_register.setToolTip(
            "Cargar .e57 y registrar múltiples scans automáticamente\n"
            "(RANSAC + ICP + Pose graph optimization)\n"
            "Atajo: Ctrl+R"
        )
        self._act_register.triggered.connect(self._on_register_scans)
        tb.addAction(self._act_register)


        # --- Cargar Fotogrametría ---
        self._act_load_photo = QAction("📷  Cargar Fotogrametría", self)
        self._act_load_photo.setShortcut(QKeySequence("Ctrl+F"))
        self._act_load_photo.setToolTip(
            "Cargar nube de fotogrametría (.ply)\n"
            "Atajo: Ctrl+F"
        )
        self._act_load_photo.triggered.connect(self._on_load_photo)
        tb.addAction(self._act_load_photo)

        tb.addSeparator()

        # --- Vistas de cámara ---
        act_iso = QAction("⬡  Isométrica", self)
        act_iso.triggered.connect(self.viewer.set_view_isometric)
        tb.addAction(act_iso)

        act_top = QAction("⬆  Planta", self)
        act_top.triggered.connect(self.viewer.set_view_top)
        tb.addAction(act_top)

        act_front = QAction("⬛  Frente", self)
        act_front.triggered.connect(self.viewer.set_view_front)
        tb.addAction(act_front)

        # Cambiar la vista con un lazo a medio dibujar mezclaría vértices de
        # cámaras distintas: se bloquean mientras el lazo está activo.
        self._acts_vista = (act_iso, act_top, act_front)

        tb.addSeparator()

        # --- Recortar nube (menú desplegable Caja / Lazo) ---
        self._btn_crop = QToolButton()
        self._btn_crop.setText("✂  Recortar")
        self._btn_crop.setToolTip("Herramientas de recorte de la nube")
        self._btn_crop.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._btn_crop.setEnabled(False)

        menu_crop = QMenu(self._btn_crop)
        grupo = QActionGroup(self)
        grupo.setExclusionPolicy(QActionGroup.ExclusionPolicy.ExclusiveOptional)
        self._act_tool_caja = QAction("⬜  Caja", self, checkable=True)
        self._act_tool_caja.setData("caja")
        self._act_tool_lazo = QAction("➰  Lazo", self, checkable=True)
        self._act_tool_lazo.setData("lazo")
        for act in (self._act_tool_caja, self._act_tool_lazo):
            grupo.addAction(act)
            menu_crop.addAction(act)
            act.triggered.connect(self._on_tool_action)
        self._btn_crop.setMenu(menu_crop)
        tb.addWidget(self._btn_crop)

        tb.addSeparator()

        # --- Limpiar escena ---
        act_clear = QAction("🗑  Limpiar", self)
        act_clear.setToolTip("Limpiar escena y proyecto actual")
        act_clear.triggered.connect(self._on_clear)
        tb.addAction(act_clear)

    def _setup_central(self):
        """Crea el layout central: viewer + panel de info en un splitter."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Viewer 3D (toma la mayor parte del espacio)
        self.viewer = Viewer3D(self)
        self.viewer.points_displayed.connect(self._on_points_displayed)
        splitter.addWidget(self.viewer)

        # Panel de información
        self.info_panel = InfoPanel(self)
        splitter.addWidget(self.info_panel)

        # El viewer ocupa ~80% del ancho
        splitter.setSizes([1000, 250])
        layout.addWidget(splitter)

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
            QSplitter::handle {
                background-color: #444444;
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

    def _on_register_scans(self):
        """Carga un .e57 y registra sus scans automáticamente."""
        from app.core.io import FILTER_STRING
        last_dir = self.settings.value("io/last_dir", os.path.expanduser("~"))
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo .e57 para registro",
            last_dir, "E57 (*.e57);;Todos los archivos (*)"
        )
        if not path:
            return

        self.settings.setValue("io/last_dir", os.path.dirname(path))
        self.show_status("Iniciando registro de scans (puede tardar varios minutos)...")
        self._set_loading(True)

        from app.core.workers import RegistrationWorker
        worker = RegistrationWorker(path, parent=self)
        worker.progress.connect(self._progress.setValue)
        worker.status.connect(self.show_status)
        worker.finished.connect(
            lambda pcd, p: self._on_cloud_loaded(pcd, p, "lidar")
        )
        worker.error.connect(self._on_load_error)
        self._current_worker = worker
        worker.start()

    def _on_load_lidar(self):
        """Carga LiDAR: un .e57 con varios scans se abre como capas por scan;
        cualquier otro archivo (o .e57 de un scan) como nube única."""
        path = self._open_file_dialog("Cargar nube LiDAR")
        if not path:
            return
        if _e57_scan_count_safe(path) >= 2:
            self._cargar_scans_por_capas(path)
        else:
            self._cargar_nube_unica(path, "lidar")

    def _cargar_nube_unica(self, path: str, cloud_type: str) -> None:
        self._load_cloud(path, cloud_type=cloud_type)

    def _on_load_photo(self):
        """Abre diálogo para cargar nube de fotogrametría."""
        path = self._open_file_dialog("Cargar nube de fotogrametría")
        if not path:
            return
        self._load_cloud(path, cloud_type="photo")

    def _cargar_scans_por_capas(self, path: str) -> None:
        """Carga el .e57 como una capa por scan (con caché de dos resoluciones)."""
        cache_dir = Path("output/scans_cache") / Path(path).stem
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
            CloudLayer(name=f"Scan {s.index:02d} ({s.n_pts_fino:,} pts)",
                       pcd=s.pcd_grueso, fine_path=s.fine_path)
            for s in scans
        ]
        stack.active_index = 0
        self._layer_stack = stack
        self.viewer.show_layers(stack)
        dock = self._ensure_crop_dock()
        dock.refresh_layers(stack)
        dock.show()
        self._btn_crop.setEnabled(True)
        self.setWindowTitle(f"{self.APP_NAME} v{self.VERSION} — scans")
        self.show_status(f"{len(scans)} scans cargados como capas. "
                         "Apaga los interiores y usa 'Exportar visibles'.")

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
            self._btn_crop.setEnabled(False)
            self.setWindowTitle(f"{self.APP_NAME} v{self.VERSION} — Sin título")
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

    def _load_cloud(self, path: str, cloud_type: str):
        """
        Lanza el worker de carga en background.
        cloud_type: "lidar" | "photo"
        """
        name = "LiDAR" if cloud_type == "lidar" else "Fotogrametría"
        self.show_status(f"Cargando {name}...")
        self._set_loading(True)

        worker = LoadWorker(path, parent=self)
        worker.progress.connect(self._progress.setValue)
        worker.finished.connect(
            lambda pcd, p: self._on_cloud_loaded(pcd, p, cloud_type)
        )
        worker.error.connect(self._on_load_error)

        # Guardar referencia para evitar que el GC destruya el worker
        self._current_worker = worker
        worker.start()

    def _on_cloud_loaded(self, pcd, path: str, cloud_type: str):
        """Callback cuando el worker termina de cargar la nube."""
        self._set_loading(False)
        actor_name = f"cloud_{cloud_type}"

        if cloud_type == "lidar":
            self.project.set_lidar(pcd, path)
            self.viewer.show_cloud(pcd, name=actor_name, point_size=2.0)
        else:
            self.project.set_photo(pcd, path)
            self.viewer.show_cloud(pcd, name=actor_name, point_size=1.5)

        self.info_panel.update_from_project(self.project)
        self.setWindowTitle(
            f"{self.APP_NAME} v{self.VERSION} — {self.project.name}"
        )
        self.show_status(f"Cargado: {path}")
        self._btn_crop.setEnabled(True)
        # Nube nueva: el stack de capas se reconstruye al activar una herramienta
        self._layer_stack = None

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
        self._act_load_photo.setEnabled(not loading)
        self._act_register.setEnabled(not loading)
        if loading:
            self._progress.setValue(0)
        QApplication.processEvents()

    # ------------------------------------------------------------------ #
    # Recorte interactivo                                                  #
    # ------------------------------------------------------------------ #

    def _ensure_crop_dock(self) -> CropDock:
        if self._crop_dock is None:
            dock = CropDock(self)
            dock.tool_apply.connect(self._on_box_apply)
            dock.tool_cancel.connect(self._on_tool_cancel)
            dock.lasso_started.connect(self._on_lasso_started)
            dock.lasso_apply.connect(self._on_lasso_apply)
            dock.lasso_cancel.connect(self._on_lasso_cancel)
            dock.layer_visibility_changed.connect(self._on_layer_visibility)
            dock.layer_activated.connect(self._on_layer_activated)
            dock.layer_removed.connect(self._on_layer_removed)
            dock.layer_restore_requested.connect(self._on_layer_restore)
            dock.export_requested.connect(self._on_export_visible)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
            self._crop_dock = dock
        return self._crop_dock

    def _ensure_layer_stack(self) -> bool:
        """Crea el stack de capas desde la nube activa si aún no existe."""
        source = self.project.fused_cloud or self.project.lidar_cloud
        if source is None:
            return False
        if self._layer_stack is None:
            self._layer_stack = LayerStack()
            self._layer_stack.reset(source)
            self.viewer.show_layers(self._layer_stack)
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
            for act in (self._act_tool_caja, self._act_tool_lazo):
                act.setChecked(False)
            return
        self._deactivate_tools(keep_checked=tool)
        dock = self._ensure_crop_dock()
        dock.set_tool(tool)
        dock.refresh_layers(self._layer_stack)
        dock.show()
        self._active_tool = tool
        if tool == "caja":
            self.viewer.start_crop_widget(
                self._layer_stack.active.pcd,
                callback=self._on_crop_bounds_changed,
            )
            self.show_status("Ajusta la caja: VERDE se conserva, ROJO se elimina.")
        else:
            self.show_status(
                "Elige la operacion y pulsa 'Iniciar lazo' en el panel Recorte."
            )

    def _deactivate_tools(self, keep_checked: str | None = None) -> None:
        """Detiene caja y lazo, limpia previews y desmarca las acciones del menú."""
        for act in self._acts_vista:
            act.setEnabled(True)
        self.viewer.stop_crop_widget()
        self.viewer.stop_lasso()
        self.viewer.clear_preview()
        self._lasso_mask = None
        self._lasso_ops = []
        self._crop_min = None
        self._crop_max = None
        self._active_tool = None
        if self._crop_dock is not None:
            self._crop_dock.set_lasso_active(False)
            self._crop_dock.set_lasso_has_selection(False)
        for act in (self._act_tool_caja, self._act_tool_lazo):
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
        fuente_nombre = stack.active.name
        try:
            recorte, descarte = stack.split_active_fino(
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
        self.viewer.show_layers(stack)
        dock = self._ensure_crop_dock()
        dock.refresh_layers(stack)
        dock.set_lasso_active(False)
        dock.set_lasso_has_selection(False)

        if self._active_tool == "caja":
            self.viewer.start_crop_widget(
                stack.active.pcd, callback=self._on_crop_bounds_changed
            )

        self.show_status(
            f"'{fuente_nombre}' quedó oculta e intacta. Nueva capa activa "
            f"'{recorte.name}' ({len(recorte.pcd.points):,} pts); "
            f"'{descarte.name}' ({len(descarte.pcd.points):,} pts) oculta."
        )

    # ---------- herramienta caja ---------- #

    def _box_keep_mask(self) -> np.ndarray | None:
        if self._layer_stack is None or self._crop_min is None:
            return None
        pts = np.asarray(self._layer_stack.active.pcd.points)
        return np.all((pts >= self._crop_min) & (pts <= self._crop_max), axis=1)

    def _on_crop_bounds_changed(
        self, min_bound: np.ndarray, max_bound: np.ndarray
    ) -> None:
        """Callback del box widget: preview verde (dentro) / rojo (fuera)."""
        self._crop_min = min_bound
        self._crop_max = max_bound
        stack = self._layer_stack
        keep = self._box_keep_mask()
        if stack is None or keep is None:
            return
        self.viewer.preview_split(
            stack.active.pcd, keep, f"layer_{stack.active_index}"
        )

    def _on_box_apply(self) -> None:
        keep = self._box_keep_mask()
        if keep is None:
            return
        self._apply_split(keep)

    def _on_tool_cancel(self) -> None:
        self._deactivate_tools()
        self.show_status("Herramienta de recorte cancelada.")

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
        self._layer_stack.set_visible(i, visible)
        self.viewer.show_layers(self._layer_stack)

    def _on_layer_activated(self, i: int) -> None:
        stack = self._layer_stack
        if stack is None or i == stack.active_index:
            return
        stack.set_active(i)
        self._lasso_mask = None
        self.viewer.stop_lasso()
        self.viewer.clear_preview()
        dock = self._ensure_crop_dock()
        dock.set_lasso_active(False)
        dock.set_lasso_has_selection(False)
        self.viewer.show_layers(stack)
        if self._active_tool == "caja":
            self.viewer.start_crop_widget(
                stack.active.pcd, callback=self._on_crop_bounds_changed
            )
        self.show_status(f"Capa activa: '{stack.active.name}'.")

    def _on_layer_removed(self, i: int) -> None:
        stack = self._layer_stack
        if stack is None:
            return
        capa = stack.layers[i]
        n = len(capa.pcd.points)
        reply = QMessageBox.question(
            self,
            "Eliminar capa",
            f"¿Eliminar la capa '{capa.name}' ({n:,} puntos)?\n"
            "Podrás deshacerlo con 'Restaurar eliminada' mientras no elimines otra.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            stack.remove(i)
        except ValueError as e:
            self.show_status(str(e))
            return
        self._removed_layer_backup = (i, capa)
        self._lasso_mask = None
        self.viewer.clear_preview()
        self.viewer.show_layers(stack)
        dock = self._ensure_crop_dock()
        dock.refresh_layers(stack)
        dock.set_restore_available(True)
        self.show_status(f"Capa '{capa.name}' eliminada (puedes restaurarla).")

    def _on_layer_restore(self) -> None:
        stack = self._layer_stack
        backup = getattr(self, "_removed_layer_backup", None)
        if stack is None or backup is None:
            return
        i, capa = backup
        stack.insert_layer(i, capa)
        self._removed_layer_backup = None
        self.viewer.show_layers(stack)
        dock = self._ensure_crop_dock()
        dock.refresh_layers(stack)
        dock.set_restore_available(False)
        self.show_status(f"Capa '{capa.name}' restaurada.")

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
