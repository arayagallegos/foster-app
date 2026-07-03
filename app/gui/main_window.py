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

import numpy as np
import open3d as o3d
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout,
    QSplitter, QToolBar, QStatusBar,
    QProgressBar, QLabel, QFileDialog,
    QMessageBox, QApplication
)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QAction, QKeySequence

from app.core.project import Project
from app.core.workers import LoadWorker
from app.gui.viewer import Viewer3D
from app.gui.panels.info_panel import InfoPanel
from app.gui.panels.crop_panel import CropPanel
from app.modules.processing import crop_cloud


class MainWindow(QMainWindow):
    """Ventana principal de Foster App."""

    APP_NAME = "Foster App"
    VERSION  = "0.1.0"

    def __init__(self):
        super().__init__()
        self.project = Project()
        self.settings = QSettings("LabPatrimonio", "FosterApp")
        self._crop_panel: CropPanel | None = None
        self._crop_min: np.ndarray | None = None
        self._crop_max: np.ndarray | None = None
        self._lasso_source: o3d.geometry.PointCloud | None = None
        self._lasso_mask: np.ndarray | None = None
        self._lasso_source_actor: str = "cloud_lidar"
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

        tb.addSeparator()

        # --- Recortar nube ---
        self._act_crop = QAction("✂  Recortar", self)
        self._act_crop.setToolTip("Recortar nube a una zona de interes")
        self._act_crop.setEnabled(False)
        self._act_crop.triggered.connect(self._on_crop)
        tb.addAction(self._act_crop)

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
        """Abre diálogo para cargar nube LiDAR."""
        path = self._open_file_dialog("Cargar nube LiDAR")
        if not path:
            return
        self._load_cloud(path, cloud_type="lidar")

    def _on_load_photo(self):
        """Abre diálogo para cargar nube de fotogrametría."""
        path = self._open_file_dialog("Cargar nube de fotogrametría")
        if not path:
            return
        self._load_cloud(path, cloud_type="photo")

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
            if self._crop_panel is not None:
                self.viewer.stop_crop_widget()
                self._crop_panel.hide()
                self._crop_panel = None
            self._crop_min = None
            self._crop_max = None
            self._lasso_source = None
            self._lasso_mask = None
            self.viewer.stop_lasso()
            self.viewer.clear()
            self.project.clear()
            self.info_panel.update_from_project(self.project)
            self._points_label.setText("")
            self._act_crop.setEnabled(False)
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
        self._act_crop.setEnabled(True)

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

    def _on_crop(self) -> None:
        """Activa el modo recorte: muestra el panel flotante y el box widget."""
        source_cloud = self.project.fused_cloud or self.project.lidar_cloud
        if source_cloud is None:
            return

        source_info = self.project.fused_info or self.project.lidar_info
        source_name = source_info.name if source_info else "nube activa"

        # Inicializar bounds con el bounding box completo de la nube
        pts = np.asarray(source_cloud.points)
        self._crop_min = pts.min(axis=0)
        self._crop_max = pts.max(axis=0)

        if self._crop_panel is None:
            self._crop_panel = CropPanel(source_name, parent=self)
            self._crop_panel.apply_requested.connect(self._on_crop_apply)
            self._crop_panel.cancel_requested.connect(self._on_crop_cancel)
            self._crop_panel.export_requested.connect(self._on_crop_export)
            self._crop_panel.visibility_changed.connect(self._on_crop_visibility)
            self._crop_panel.lasso_started.connect(self._on_lasso_started)
            self._crop_panel.lasso_apply_requested.connect(self._on_lasso_apply)
            self._crop_panel.lasso_cancel_requested.connect(self._on_lasso_cancel)

        viewer_pos = self.viewer.mapTo(self, self.viewer.rect().topLeft())
        self._crop_panel.move(viewer_pos.x() + 10, viewer_pos.y() + 10)
        self._crop_panel.show()
        self._crop_panel.raise_()

        self.viewer.start_crop_widget(
            source_cloud,
            callback=self._on_crop_bounds_changed,
        )
        self.show_status("Ajusta la caja en el viewer y presiona Aplicar.")

    def _on_crop_bounds_changed(
        self,
        min_bound: np.ndarray,
        max_bound: np.ndarray,
    ) -> None:
        """Callback del viewer: actualiza los bounds actuales del widget."""
        self._crop_min = min_bound
        self._crop_max = max_bound

    def _on_crop_apply(self) -> None:
        """Aplica el recorte con los bounds actuales del widget."""
        source_cloud = self.project.fused_cloud or self.project.lidar_cloud
        if source_cloud is None or self._crop_min is None:
            return

        try:
            result = crop_cloud(source_cloud, self._crop_min, self._crop_max)
        except ValueError:
            self.show_status("La seleccion no contiene puntos. Ajusta la caja.")
            return

        self.project.set_cropped(result)
        self.viewer.show_cloud(result, name="cloud_cropped", point_size=2.0, replace=True)
        self.viewer.set_actor_visibility("cloud_lidar", True)

        if self._crop_panel is not None:
            self._crop_panel.show_visibility_controls()

        n = len(result.points)
        self.show_status(
            f"Recorte aplicado: {n:,} puntos. "
            "Ajusta la caja o presiona Cancelar para salir."
        )

    def _on_crop_cancel(self) -> None:
        """Cancela el modo recorte y cierra el panel."""
        self.viewer.stop_crop_widget()
        if self._crop_panel is not None:
            self._crop_panel.hide()
            self._crop_panel = None
        self._crop_min = None
        self._crop_max = None
        self.show_status("Recorte cancelado.")

    def _on_crop_visibility(self, show_original: bool, show_cropped: bool) -> None:
        """Muestra u oculta las nubes segun los checkboxes del panel."""
        self.viewer.set_actor_visibility("cloud_lidar", show_original)
        self.viewer.set_actor_visibility("cloud_cropped", show_cropped)

    def _on_crop_export(self) -> None:
        """Abre dialogo para guardar la nube recortada como .ply."""
        from app.core.io import export_point_cloud
        pcd = self.project.cropped_cloud
        if pcd is None:
            return
        last_dir = self.settings.value("io/last_dir", os.path.expanduser("~"))
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar nube recortada", last_dir, "PLY (*.ply)"
        )
        if not path:
            return
        if not path.lower().endswith(".ply"):
            path += ".ply"
        try:
            export_point_cloud(pcd, path)
            self.settings.setValue("io/last_dir", os.path.dirname(path))
            self.show_status(f"Nube recortada exportada: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error al exportar", str(e))

    def _on_lasso_started(self, set_op: str) -> None:
        source = self.project.cropped_cloud or self.project.lidar_cloud
        if source is None:
            return
        self._lasso_source = source
        self._lasso_source_actor = (
            "cloud_cropped" if self.project.cropped_cloud is not None else "cloud_lidar"
        )
        self._lasso_set_op = set_op
        if self._lasso_mask is None:
            self._lasso_mask = np.zeros(len(source.points), dtype=bool)

        if self._crop_panel is not None:
            self._crop_panel.set_lasso_active(True)

        self.viewer.start_lasso(self._on_lasso_polygon_closed)
        self.show_status(
            "Haz clic para agregar vertices. "
            "Clic derecho o doble clic para cerrar el lazo."
        )

    def _on_lasso_polygon_closed(self, verts: list[tuple[int, int]]) -> None:
        if self._lasso_source is None:
            return

        from app.modules.processing import apply_lasso

        screen_pts, valid = self.viewer.project_cloud_to_screen(self._lasso_source)
        try:
            new_mask = apply_lasso(
                screen_pts, valid, verts, self._lasso_set_op, self._lasso_mask
            )
        except ValueError as e:
            self.show_status(str(e))
            if self._crop_panel is not None:
                self._crop_panel.set_lasso_active(False)
            return

        if not new_mask.any():
            self.show_status("La operacion no selecciono puntos. Mascara sin cambios.")
        else:
            self._lasso_mask = new_mask
            n = int(new_mask.sum())
            self.show_status(f"Lazo: {n:,} puntos seleccionados.")

        if self._crop_panel is not None:
            self._crop_panel.set_lasso_active(False)
            self._crop_panel.set_lasso_has_selection(
                self._lasso_mask is not None and self._lasso_mask.any()
            )

        self.viewer.highlight_selection(
            self._lasso_source, self._lasso_mask, self._lasso_source_actor
        )

    def _on_lasso_apply(self) -> None:
        if self._lasso_source is None or self._lasso_mask is None:
            return
        if not self._lasso_mask.any():
            self.show_status("No hay puntos seleccionados.")
            return

        pts = np.asarray(self._lasso_source.points)[self._lasso_mask]
        result = o3d.geometry.PointCloud()
        result.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

        if self._lasso_source.has_colors():
            cols = np.asarray(self._lasso_source.colors)[self._lasso_mask]
            result.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

        self.project.set_cropped(result)
        self.viewer.stop_lasso()
        self.viewer.show_cloud(result, name="cloud_cropped", point_size=2.0, replace=True)
        self.viewer.set_actor_visibility("cloud_lidar", True)

        self._lasso_source = None
        self._lasso_mask = None

        if self._crop_panel is not None:
            self._crop_panel.set_lasso_active(False)
            self._crop_panel.set_lasso_has_selection(False)
            self._crop_panel.show_visibility_controls()

        n = len(result.points)
        self.show_status(
            f"Lazo aplicado: {n:,} puntos. Exporta con 'Exportar recorte (.ply)'."
        )

    def _on_lasso_cancel(self) -> None:
        self.viewer.stop_lasso()
        self._lasso_source = None
        self._lasso_mask = None
        if self._crop_panel is not None:
            self._crop_panel.set_lasso_active(False)
            self._crop_panel.set_lasso_has_selection(False)
        self.show_status("Lazo cancelado.")

    # ------------------------------------------------------------------ #
    # Persistencia                                                         #
    # ------------------------------------------------------------------ #

    def closeEvent(self, event):
        self.settings.setValue("window/geometry", self.saveGeometry())
        super().closeEvent(event)
