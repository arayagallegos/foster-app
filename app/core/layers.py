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
    descarte: bool = False           # quedó fuera de un recorte: se puede borrar en bloque


def _subset(pcd: o3d.geometry.PointCloud, mask: np.ndarray) -> o3d.geometry.PointCloud:
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[mask])
    if pcd.has_colors():
        out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])
    return out


def _slug(nombre: str) -> str:
    """Nombre de capa → nombre de archivo seguro (los nombres traen comas,
    paréntesis y puntos que Windows no acepta o que confunden la extensión)."""
    limpio = "".join(c if c.isalnum() else "_" for c in nombre)
    return "_".join(filter(None, limpio.split("_")))[:60] or "capa"


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

    def split_active_fino(self, keep_mask_grueso, fine_keep_fn, edits_dir):
        """
        Recorte que preserva la resolución fina. Si la capa activa tiene fine_path,
        recorta también la nube fina de disco con fine_keep_fn y asigna fine_path a
        las hijas. Si no, equivale a split_active.
        `fine_keep_fn(pcd_fino) -> np.ndarray[bool]` la provee el llamador (conoce
        la cámara/caja para reaplicar el recorte a cualquier resolución).
        """
        fuente = self.active
        recorte, descarte = self.split_active(keep_mask_grueso)   # divide el grueso
        if fuente.fine_path is None or not Path(fuente.fine_path).exists():
            return recorte, descarte

        fino = o3d.io.read_point_cloud(str(fuente.fine_path))
        keep_fino = np.asarray(fine_keep_fn(fino), dtype=bool)
        edits_dir = Path(edits_dir)
        edits_dir.mkdir(parents=True, exist_ok=True)
        k = self._split_counter                                    # ya incrementado
        rec_path = edits_dir / f"recorte_{k}_fino.ply"
        des_path = edits_dir / f"descarte_{k}_fino.ply"
        o3d.io.write_point_cloud(str(rec_path), _subset(fino, keep_fino))
        o3d.io.write_point_cloud(str(des_path), _subset(fino, ~keep_fino))
        recorte.fine_path = rec_path
        descarte.fine_path = des_path
        return recorte, descarte

    def split_visible_fino(self, keep_fn, fine_keep_fn, edits_dir):
        """
        Aplica el MISMO criterio de recorte a todas las capas visibles.

        Cada capa que queda partida se reemplaza por sus dos mitades, y ambas
        heredan el nombre de origen: es lo que permite seguir sabiendo de qué
        scan viene cada punto y, por lo tanto, separar interior de exterior
        apagando capas.

        Las capas que quedan enteras de un lado NO se dividen: si todo cae
        dentro se dejan tal cual, y si todo cae fuera solo se ocultan. Con 35
        scans la mayoría cae en uno de esos dos casos, y dividirlas igual
        llenaría la lista de capas vacías.

        A diferencia de `split_active_fino`, la capa de origen no se conserva:
        con 35 capas, guardar una copia oculta de cada una triplicaría la
        memoria. No se pierde nada — la unión de las dos mitades es exactamente
        la capa original.

        `keep_fn(pcd) -> mask bool` y `fine_keep_fn(pcd_fino) -> mask bool|None`
        los provee el llamador (conoce la caja o el lazo).
        Devuelve (n_divididas, n_ocultadas, n_intactas).
        """
        visibles = [c for c in self.layers if c.visible]
        if not visibles:
            raise ValueError("Ninguna capa visible que recortar.")

        # Primero se calculan todas las máscaras y se valida: si el recorte no
        # conserva nada en ninguna capa, es un error del usuario (caja fuera de
        # la nube) y no debe dejar el stack a medio modificar.
        masks: dict[int, np.ndarray] = {}
        for i, capa in enumerate(self.layers):
            if not capa.visible:
                continue
            masks[i] = np.asarray(keep_fn(capa.pcd), dtype=bool)
        if not any(m.any() for m in masks.values()):
            raise ValueError(
                "El recorte no conserva ningún punto de las capas visibles.")

        self._split_counter += 1
        k = self._split_counter
        edits_dir = Path(edits_dir)

        nuevas: list[CloudLayer] = []
        activa = -1          # índice, no la capa: dos capas pueden compararse iguales
        n_div = n_ocul = n_intacta = 0

        for i, capa in enumerate(self.layers):
            mask = masks.get(i)
            if mask is None:                    # capa oculta: no se toca
                nuevas.append(capa)
                continue
            if mask.all():                      # entera dentro
                if activa < 0:
                    activa = len(nuevas)
                nuevas.append(capa)
                n_intacta += 1
                continue
            if not mask.any():                  # entera fuera
                capa.visible = False
                capa.descarte = True
                nuevas.append(capa)
                n_ocul += 1
                continue

            dentro = CloudLayer(name=f"{capa.name} · dentro {k}",
                                pcd=_subset(capa.pcd, mask))
            fuera = CloudLayer(name=f"{capa.name} · fuera {k}",
                               pcd=_subset(capa.pcd, ~mask), visible=False,
                               descarte=True)
            if capa.fine_path is not None and Path(capa.fine_path).exists():
                fino = o3d.io.read_point_cloud(str(capa.fine_path))
                keep_f = fine_keep_fn(fino)
                if keep_f is not None:
                    keep_f = np.asarray(keep_f, dtype=bool)
                    edits_dir.mkdir(parents=True, exist_ok=True)
                    base = _slug(capa.name)
                    p_in = edits_dir / f"{base}_dentro_{k}.ply"
                    p_out = edits_dir / f"{base}_fuera_{k}.ply"
                    o3d.io.write_point_cloud(str(p_in), _subset(fino, keep_f))
                    o3d.io.write_point_cloud(str(p_out), _subset(fino, ~keep_f))
                    dentro.fine_path, fuera.fine_path = p_in, p_out
            if activa < 0:
                activa = len(nuevas)
            nuevas.extend((dentro, fuera))
            n_div += 1

        self.layers = nuevas
        self.active_index = max(activa, 0)
        return n_div, n_ocul, n_intacta

    def unir(self, indices, nombre: str) -> CloudLayer:
        """Funde varias capas en una sola, que reemplaza a la primera.

        Es la operación inversa del recorte, y sin ella el flujo de DBSCAN no
        cierra: una entidad como la cúpula aparece partida en decenas de
        clusters, y capturar cada uno por separado dejaría decenas de capas sin
        manera de recomponer la entidad.

        Las capas fuente desaparecen: la unión contiene exactamente sus puntos,
        así que conservarlas solo duplicaría memoria.
        """
        indices = sorted(set(int(i) for i in indices))
        if len(indices) < 2:
            raise ValueError("Hay que elegir al menos dos capas para unir.")
        if not all(0 <= i < len(self.layers) for i in indices):
            raise IndexError("Alguna de las capas indicadas no existe.")
        nombre = str(nombre).strip()
        if not nombre:
            raise ValueError("La capa unida necesita un nombre.")

        origen = [self.layers[i] for i in indices]
        pts = np.vstack([np.asarray(c.pcd.points) for c in origen])
        unida = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        if all(c.pcd.has_colors() for c in origen):
            unida.colors = o3d.utility.Vector3dVector(
                np.vstack([np.asarray(c.pcd.colors) for c in origen]))

        capa = CloudLayer(name=nombre, pcd=unida, visible=True)
        destino = indices[0]
        self.layers = [
            capa if i == destino else c
            for i, c in enumerate(self.layers) if i == destino or i not in indices
        ]
        self.active_index = self.layers.index(capa)
        return capa

    def contar_descartes(self) -> int:
        return sum(1 for c in self.layers if c.descarte)

    def eliminar_descartes(self) -> int:
        """
        Borra de una vez todas las capas que quedaron fuera de algún recorte.

        Sin esto hay que eliminarlas una por una, y cada recorte sobre 35 scans
        puede dejar decenas. Devuelve cuántas se eliminaron.
        """
        activa = self.layers[self.active_index] if self.layers else None
        quedan = [c for c in self.layers if not c.descarte]
        if not quedan:
            raise ValueError(
                "Todas las capas son descartes: eliminarlas dejaría el proyecto vacío.")

        n = len(self.layers) - len(quedan)
        self.layers = quedan
        # la capa activa pudo ser un descarte; en ese caso se cae a la primera
        self.active_index = next(
            (i for i, c in enumerate(quedan) if c is activa), 0)
        return n

    def renombrar(self, i: int, nombre: str) -> None:
        """Renombra una capa. El nombre es lo que después identifica la entidad
        al exportar, así que no puede quedar vacío."""
        nombre = str(nombre).strip()
        if not nombre:
            raise ValueError("El nombre de la capa no puede estar vacío.")
        if not 0 <= i < len(self.layers):
            raise IndexError(f"No existe la capa {i}.")
        self.layers[i].name = nombre

    def mostrar_solo(self, indices) -> None:
        """Deja visibles únicamente las capas indicadas.

        Con 35 capas-scan, aislar una es la operación más frecuente y hacerlo a
        mano exige 34 clics.
        """
        indices = {int(i) for i in indices}
        if not indices:
            raise ValueError("Hay que indicar al menos una capa.")
        if not all(0 <= i < len(self.layers) for i in indices):
            raise IndexError("Alguna de las capas indicadas no existe.")
        for i, capa in enumerate(self.layers):
            capa.visible = i in indices

    def set_visibles(self, indices, visible: bool) -> None:
        """Cambia la visibilidad de varias capas de una vez."""
        for i in indices:
            if not 0 <= int(i) < len(self.layers):
                raise IndexError(f"No existe la capa {i}.")
            self.layers[int(i)].visible = bool(visible)

    def invertir_visibilidad(self) -> None:
        """Muestra lo oculto y oculta lo visible.

        Es el atajo natural cuando se quiere el complemento de lo elegido: por
        ejemplo, pasar de mirar los scans exteriores a mirar los interiores.
        """
        for capa in self.layers:
            capa.visible = not capa.visible

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

    def merge_visible_fine(self) -> o3d.geometry.PointCloud:
        """
        Como merge_visible, pero usa el .ply fino de disco (fine_path) cuando existe.
        Para capas sin fine_path usa los puntos en memoria.
        """
        visibles = [c for c in self.layers if c.visible]
        if not visibles:
            raise ValueError("Ninguna capa visible que exportar.")
        pcds = []
        for c in visibles:
            if c.fine_path is not None and Path(c.fine_path).exists():
                pcds.append(o3d.io.read_point_cloud(str(c.fine_path)))
            else:
                pcds.append(c.pcd)
        pts = np.vstack([np.asarray(p.points) for p in pcds])
        out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        if all(p.has_colors() for p in pcds):
            cols = np.vstack([np.asarray(p.colors) for p in pcds])
            out.colors = o3d.utility.Vector3dVector(cols)
        return out

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)
