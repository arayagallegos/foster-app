"""
project.py — Estado global de la aplicación Foster.

El objeto Project es el modelo de datos central: guarda todas las nubes,
mallas y segmentos cargados, y mantiene un stack de historial para undo.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import copy

import numpy as np
import open3d as o3d


@dataclass
class CloudInfo:
    """Metadatos de una nube de puntos cargada."""
    name: str
    path: str
    n_points_original: int
    n_points_display: int
    has_colors: bool
    bbox_min: np.ndarray
    bbox_max: np.ndarray

    @property
    def dimensions(self) -> np.ndarray:
        """Dimensiones del bounding box en metros."""
        return self.bbox_max - self.bbox_min

    @property
    def summary(self) -> str:
        d = self.dimensions
        return (
            f"Puntos: {self.n_points_original:,} "
            f"(mostrando: {self.n_points_display:,})\n"
            f"Dimensiones: {d[0]:.2f} x {d[1]:.2f} x {d[2]:.2f} m\n"
            f"Color RGB: {'Sí' if self.has_colors else 'No (coloreado por altura)'}"
        )


@dataclass
class ProjectState:
    """
    Snapshot del estado del proyecto en un momento dado.
    Se guarda en el stack de historial para poder hacer undo.
    """
    lidar_cloud: Optional[o3d.geometry.PointCloud] = None
    photo_cloud: Optional[o3d.geometry.PointCloud] = None
    fused_cloud: Optional[o3d.geometry.PointCloud] = None
    mesh: Optional[o3d.geometry.TriangleMesh] = None
    segments: dict = field(default_factory=dict)
    cropped_cloud: Optional[o3d.geometry.PointCloud] = None


class Project:
    """
    Estado global de la aplicación. Una sola instancia por sesión.

    Centraliza todas las nubes de puntos, mallas y segmentos.
    Implementa undo mediante un stack de estados.
    """

    MAX_HISTORY = 10  # máximo de estados guardados en historial

    def __init__(self):
        self._state = ProjectState()
        self._history: list[ProjectState] = []

        # Metadatos de las nubes cargadas (para el panel de info)
        self.lidar_info: Optional[CloudInfo] = None
        self.photo_info: Optional[CloudInfo] = None
        self.fused_info: Optional[CloudInfo] = None
        self.cropped_info: Optional[CloudInfo] = None

        # Nombre del proyecto (nombre del primer archivo cargado)
        self.name: str = "Sin título"

    # ------------------------------------------------------------------ #
    # Propiedades de acceso a las nubes                                    #
    # ------------------------------------------------------------------ #

    @property
    def lidar_cloud(self) -> Optional[o3d.geometry.PointCloud]:
        return self._state.lidar_cloud

    @property
    def photo_cloud(self) -> Optional[o3d.geometry.PointCloud]:
        return self._state.photo_cloud

    @property
    def fused_cloud(self) -> Optional[o3d.geometry.PointCloud]:
        return self._state.fused_cloud

    @property
    def cropped_cloud(self) -> Optional[o3d.geometry.PointCloud]:
        return self._state.cropped_cloud

    @property
    def mesh(self) -> Optional[o3d.geometry.TriangleMesh]:
        return self._state.mesh

    @property
    def segments(self) -> dict:
        return self._state.segments

    @property
    def has_any_cloud(self) -> bool:
        return any([
            self._state.lidar_cloud is not None,
            self._state.photo_cloud is not None,
            self._state.fused_cloud is not None,
        ])

    # ------------------------------------------------------------------ #
    # Métodos de modificación (guardan historial automáticamente)         #
    # ------------------------------------------------------------------ #

    def set_lidar(self, pcd: o3d.geometry.PointCloud, path: str):
        self._save_history()
        self._state.lidar_cloud = pcd
        self.lidar_info = _make_cloud_info(pcd, path)
        if self.name == "Sin título":
            self.name = Path(path).stem

    def set_photo(self, pcd: o3d.geometry.PointCloud, path: str):
        self._save_history()
        self._state.photo_cloud = pcd
        self.photo_info = _make_cloud_info(pcd, path)

    def set_fused(self, pcd: o3d.geometry.PointCloud):
        self._save_history()
        self._state.fused_cloud = pcd
        self.fused_info = _make_cloud_info(pcd, "fusionada")

    def set_cropped(self, pcd: o3d.geometry.PointCloud) -> None:
        self._save_history()
        self._state.cropped_cloud = pcd
        self.cropped_info = _make_cloud_info(pcd, "recortada")

    def set_mesh(self, mesh: o3d.geometry.TriangleMesh):
        self._save_history()
        self._state.mesh = mesh

    def add_segment(self, name: str, pcd: o3d.geometry.PointCloud):
        self._save_history()
        self._state.segments[name] = pcd

    def clear(self):
        """Reinicia el proyecto completamente."""
        self._save_history()
        self._state = ProjectState()
        self.lidar_info = None
        self.photo_info = None
        self.fused_info = None
        self.cropped_info = None
        self.name = "Sin título"

    # ------------------------------------------------------------------ #
    # Historial / Undo                                                     #
    # ------------------------------------------------------------------ #

    def _save_history(self):
        """Guarda una copia del estado actual en el stack de historial."""
        self._history.append(copy.deepcopy(self._state))
        if len(self._history) > self.MAX_HISTORY:
            self._history.pop(0)

    def undo(self) -> bool:
        """
        Restaura el estado anterior.
        Retorna True si había historial, False si no.
        """
        if not self._history:
            return False
        self._state = self._history.pop()
        return True

    @property
    def can_undo(self) -> bool:
        return len(self._history) > 0


# ------------------------------------------------------------------ #
# Funciones auxiliares                                                #
# ------------------------------------------------------------------ #

def _make_cloud_info(pcd: o3d.geometry.PointCloud, path: str) -> CloudInfo:
    """Construye el objeto CloudInfo desde una nube Open3D."""
    pts = np.asarray(pcd.points)
    n = len(pts)
    has_colors = pcd.has_colors()
    bbox_min = pts.min(axis=0) if n > 0 else np.zeros(3)
    bbox_max = pts.max(axis=0) if n > 0 else np.zeros(3)

    # n_points_display se actualiza desde el viewer cuando hace downsample
    return CloudInfo(
        name=Path(path).name,
        path=str(path),
        n_points_original=n,
        n_points_display=n,
        has_colors=has_colors,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )
