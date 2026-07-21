"""
layers.py — Modelo de capas para el recorte iterativo de nubes de puntos.

Lógica pura (sin Qt): MainWindow es dueño del LayerStack y el viewer/dock solo
lo leen. Cada recorte divide la capa activa en "lo que queda" y una capa de
descarte oculta — nada se pierde hasta que el usuario elimina una capa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d


@dataclass
class CloudLayer:
    name: str
    pcd: o3d.geometry.PointCloud
    visible: bool = True
    fine_path: Path | None = None    # versión fina en disco (capas-scan); None = normal


def _subset(pcd: o3d.geometry.PointCloud, mask: np.ndarray) -> o3d.geometry.PointCloud:
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[mask])
    if pcd.has_colors():
        out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    return out


@dataclass
class LayerStack:
    layers: list[CloudLayer] = field(default_factory=list)
    active_index: int = -1
    _split_counter: int = 0

    def reset(self, pcd: o3d.geometry.PointCloud, name: str = "Original") -> None:
        """Reinicia el stack con una única capa visible y activa."""
        self.layers = [CloudLayer(name=name, pcd=pcd)]
        self.active_index = 0
        self._split_counter = 0

    @property
    def active(self) -> CloudLayer:
        return self.layers[self.active_index]

    def split_active(self, keep_mask: np.ndarray) -> tuple[CloudLayer, CloudLayer]:
        """
        Recorte NO destructivo: la capa activa queda intacta pero se oculta, y
        se crean dos capas nuevas: "Recorte N" (lo conservado, visible y nueva
        activa) y "Descarte N" (lo eliminado, oculta).
        Devuelve (recorte, descarte).
        """
        keep_mask = np.asarray(keep_mask, dtype=bool)
        fuente = self.active
        n = len(fuente.pcd.points)
        if len(keep_mask) != n:
            raise ValueError(
                f"Máscara de largo {len(keep_mask)} para capa de {n} puntos."
            )
        if keep_mask.all():
            raise ValueError("La máscara conserva todo: no hay nada que separar.")
        if not keep_mask.any():
            raise ValueError("La máscara no conserva nada: la capa quedaría vacía.")

        self._split_counter += 1
        i = self._split_counter
        recorte = CloudLayer(name=f"Recorte {i}", pcd=_subset(fuente.pcd, keep_mask))
        descarte = CloudLayer(
            name=f"Descarte {i}", pcd=_subset(fuente.pcd, ~keep_mask), visible=False
        )
        fuente.visible = False
        self.layers.append(recorte)
        self.layers.append(descarte)
        self.active_index = len(self.layers) - 2  # la capa "Recorte N"
        return recorte, descarte

    def set_visible(self, i: int, visible: bool) -> None:
        self.layers[i].visible = bool(visible)

    def set_active(self, i: int) -> None:
        if not 0 <= i < len(self.layers):
            raise IndexError(f"No existe la capa {i}.")
        self.active_index = i

    def remove(self, i: int) -> None:
        if len(self.layers) <= 1:
            raise ValueError("No se puede eliminar la única capa.")
        del self.layers[i]
        if self.active_index >= len(self.layers) or self.active_index == i:
            self.active_index = 0
        elif self.active_index > i:
            self.active_index -= 1

    def insert_layer(self, i: int, layer: CloudLayer) -> None:
        """Reinserta una capa (p. ej. al deshacer una eliminación)."""
        i = max(0, min(i, len(self.layers)))
        self.layers.insert(i, layer)
        if self.active_index >= i:
            self.active_index += 1

    def merge_visible(self) -> o3d.geometry.PointCloud:
        """Concatena puntos (y colores si todas los tienen) de las capas visibles."""
        visibles = [c for c in self.layers if c.visible]
        if not visibles:
            raise ValueError("Ninguna capa visible que exportar.")
        pts = np.vstack([np.asarray(c.pcd.points) for c in visibles])
        out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        if all(c.pcd.has_colors() for c in visibles):
            cols = np.vstack([np.asarray(c.pcd.colors) for c in visibles])
            out.colors = o3d.utility.Vector3dVector(cols)
        return out

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)
