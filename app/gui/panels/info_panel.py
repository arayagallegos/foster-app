"""
info_panel.py — Panel lateral con información de la nube de puntos activa.

Muestra: nombre del archivo, cantidad de puntos, dimensiones del bounding box,
presencia de color RGB, y estado del proyecto.
"""

from __future__ import annotations
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox,
    QSizePolicy, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from app.core.project import Project, CloudInfo


class InfoPanel(QWidget):
    """
    Panel lateral derecho con información de la nube y estado del proyecto.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(220)
        self.setMaximumWidth(280)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        # --- Título ---
        title = QLabel("Información")
        font = QFont()
        font.setBold(True)
        font.setPointSize(10)
        title.setFont(font)
        layout.addWidget(title)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)

        # --- Grupo: Nube LiDAR ---
        self._lidar_group = QGroupBox("Nube LiDAR")
        self._lidar_layout = QVBoxLayout(self._lidar_group)
        self._lidar_label = _make_info_label("Sin cargar")
        self._lidar_layout.addWidget(self._lidar_label)
        layout.addWidget(self._lidar_group)

        # --- Grupo: Nube Fotogrametría ---
        self._photo_group = QGroupBox("Fotogrametría")
        self._photo_layout = QVBoxLayout(self._photo_group)
        self._photo_label = _make_info_label("Sin cargar")
        self._photo_layout.addWidget(self._photo_label)
        layout.addWidget(self._photo_group)

        # --- Grupo: Nube Fusionada ---
        self._fused_group = QGroupBox("Nube Fusionada")
        self._fused_layout = QVBoxLayout(self._fused_group)
        self._fused_label = _make_info_label("No generada")
        self._fused_layout.addWidget(self._fused_label)
        layout.addWidget(self._fused_group)

        # Espaciador al fondo
        layout.addStretch()

    # ------------------------------------------------------------------ #
    # Métodos de actualización                                             #
    # ------------------------------------------------------------------ #

    def update_from_project(self, project: Project):
        """Actualiza el panel con el estado actual del proyecto."""
        self._update_cloud_label(self._lidar_label, project.lidar_info)
        self._update_cloud_label(self._photo_label, project.photo_info)
        self._update_cloud_label(self._fused_label, project.fused_info)

    def update_display_count(self, n_original: int, n_displayed: int):
        """
        Actualiza el conteo de puntos mostrados en pantalla.
        Llamado por el viewer cuando hace downsample.
        """
        if n_original != n_displayed:
            extra = f"\nMostrando: {n_displayed:,} (reducido para visualización)"
        else:
            extra = ""
        # Este método se puede refinar para saber cuál nube actualizar
        # Por ahora solo agrega info al label de la última nube cargada
        _ = extra  # usado en versiones futuras

    def _update_cloud_label(self, label: QLabel, info: Optional[CloudInfo]):
        if info is None:
            label.setText("Sin cargar")
            label.setStyleSheet("color: #888888;")
            return

        d = info.dimensions
        text = (
            f"<b>{info.name}</b><br>"
            f"Puntos: {info.n_points_original:,}<br>"
            f"Dim: {d[0]:.1f} × {d[1]:.1f} × {d[2]:.1f} m<br>"
            f"RGB: {'✓' if info.has_colors else '✗ (altura)'}"
        )
        label.setText(text)
        label.setStyleSheet("color: #dddddd;")


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _make_info_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.RichText)
    label.setStyleSheet("color: #888888; padding: 2px;")
    label.setSizePolicy(
        QSizePolicy.Policy.Preferred,
        QSizePolicy.Policy.Minimum
    )
    return label
