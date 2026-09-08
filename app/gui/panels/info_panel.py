"""
info_panel.py — Panel lateral con información de la nube cargada.

Muestra solo lo que condiciona decisiones del usuario: cuántos puntos hay, de
cuántos escaneos vienen, cuánto ocupa el archivo, qué tamaño tiene la escena y
cuál es el espaciado entre puntos.

El espaciado es el dato menos evidente y el más útil: de él dependen el vóxel
del caché, el `eps` de la agrupación por densidad y la tolerancia de las
primitivas. Antes había que deducirlo probando.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox, QLabel, QVBoxLayout, QWidget,
)

from app.core.project import CloudInfo, Project


def _texto_tamano(n_bytes: int) -> str:
    for unidad in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024 or unidad == "GB":
            return f"{n_bytes:.1f} {unidad}" if unidad != "B" else f"{n_bytes} B"
        n_bytes /= 1024.0
    return f"{n_bytes:.1f} GB"


class InfoPanel(QWidget):
    """Panel lateral con los datos de la nube activa."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(220)
        self.setMaximumWidth(280)
        self._n_escaneos: Optional[int] = None
        self._espaciado: Optional[float] = None
        self._en_cache: Optional[int] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        self._grupo = QGroupBox("Nube de puntos")
        caja = QVBoxLayout(self._grupo)
        self._etiqueta = QLabel("Sin cargar")
        self._etiqueta.setWordWrap(True)
        self._etiqueta.setTextFormat(Qt.TextFormat.RichText)
        self._etiqueta.setStyleSheet("color: #cccccc; font-size: 11px;")
        caja.addWidget(self._etiqueta)
        layout.addWidget(self._grupo)

        self._grupo_vista = QGroupBox("En pantalla")
        caja_vista = QVBoxLayout(self._grupo_vista)
        self._etiqueta_vista = QLabel("Sin nube cargada")
        self._etiqueta_vista.setWordWrap(True)
        self._etiqueta_vista.setStyleSheet("color: #cccccc; font-size: 11px;")
        caja_vista.addWidget(self._etiqueta_vista)
        layout.addWidget(self._grupo_vista)

        layout.addStretch(1)

    # ------------------------------------------------------------------ #
    # API                                                                #
    # ------------------------------------------------------------------ #

    def set_scan_count(self, n: Optional[int]) -> None:
        """Nº de escaneos del archivo. None si no aplica (un `.ply` suelto)."""
        self._n_escaneos = n

    def set_puntos_en_cache(self, n: Optional[int]) -> None:
        """Puntos que quedaron tras voxelizar. Se muestra aparte del total del
        archivo: confundirlos da una idea equivocada del dato de partida."""
        self._en_cache = n

    def set_espaciado(self, espaciado: Optional[float]) -> None:
        """Distancia típica entre puntos vecinos, en metros."""
        self._espaciado = espaciado

    def update_from_project(self, project: Project) -> None:
        self._etiqueta.setText(self._describir(project.lidar_info))

    def update_display_count(self, n_original: int, n_displayed: int) -> None:
        if n_original <= 0:
            self._etiqueta_vista.setText("Sin nube cargada")
            return
        pct = 100.0 * n_displayed / n_original
        self._etiqueta_vista.setText(
            f"{n_displayed:,} de {n_original:,} puntos ({pct:.0f} %)")

    def update_layer_summary(self, capas, activas) -> None:
        """Resumen en vivo de las capas: cuánto se ve y cuánto se está usando.

        Se recalcula en cada cambio de visibilidad o de conjunto activo. Antes se
        fijaba una sola vez al cargar, así que el panel seguía anunciando dos
        millones de puntos en pantalla con todas las capas apagadas.
        """
        total = sum(len(c.pcd.points) for c in capas)
        if not capas or total <= 0:
            self._etiqueta_vista.setText("Sin nube cargada")
            return
        vis = [c for c in capas if c.visible]
        n_vis = sum(len(c.pcd.points) for c in vis)
        n_act = sum(len(c.pcd.points) for c in activas)
        self._etiqueta_vista.setText(
            f"<b>Ver:</b> {len(vis)} de {len(capas)} capas<br>"
            f"{n_vis:,} de {total:,} pts ({100.0 * n_vis / total:.0f} %)"
            f"<br><br><b>Usar:</b> {len(activas)} "
            f"{'capa' if len(activas) == 1 else 'capas'}<br>"
            f"{n_act:,} pts ({100.0 * n_act / total:.0f} %)")

    # ------------------------------------------------------------------ #

    def _describir(self, info: Optional[CloudInfo]) -> str:
        if info is None:
            return "Sin cargar"

        filas = [f"<b>{Path(info.path).name if info.path else info.name}</b>"]
        filas.append(f"Puntos del archivo: {info.n_points_original:,}")
        if self._en_cache and self._en_cache != info.n_points_original:
            pct = 100.0 * self._en_cache / max(info.n_points_original, 1)
            filas.append(f"En el caché: {self._en_cache:,} ({pct:.1f} %)")
        if self._n_escaneos:
            filas.append(f"Escaneos: {self._n_escaneos}")
        try:
            if info.path and Path(info.path).exists():
                filas.append(f"Archivo: {_texto_tamano(Path(info.path).stat().st_size)}")
        except OSError:
            pass
        d = info.dimensions
        filas.append(f"Dimensiones: {d[0]:.1f} × {d[1]:.1f} × {d[2]:.1f} m")
        if self._espaciado:
            filas.append(f"Espaciado: {self._espaciado:.3f} m")
        filas.append("Color RGB: " + ("sí" if info.has_colors else "no"))
        return "<br>".join(filas)
