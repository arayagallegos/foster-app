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
    n_pts_archivo: int | None = None  # puntos del escaneo antes de submuestrear


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
    active_indices: list[int] = field(default_factory=list)
    _split_counter: int = 0
    _cache_combinada: tuple | None = field(default=None, repr=False, compare=False)

    def reset(self, pcd: o3d.geometry.PointCloud, name: str = "Original") -> None:
        """Reinicia el stack con una única capa visible y activa."""
        # Sin submuestreo de por medio, el archivo y el dato de trabajo son el
        # mismo: decirlo evita que la columna muestre un guion como si el dato
        # faltara.
        self.layers = [CloudLayer(name=name, pcd=pcd,
                                  n_pts_archivo=len(pcd.points))]
        self.active_index = 0
        self._split_counter = 0

    # ------------------------------------------------------------------ activa
    # Puede haber VARIAS capas activas a la vez. Es lo que espera quien usa la
    # herramienta: si una entidad quedó repartida entre seis escaneos, obligar a
    # tratarlos de a uno es arbitrario. Las herramientas ven la unión de sus
    # puntos, y el resultado se reparte de vuelta a la capa de la que vino cada
    # punto, para no perder de qué escaneo procede.

    @property
    def active_index(self) -> int:
        """La primera capa activa.

        Se conserva porque varias operaciones necesitan una sola referencia:
        dónde insertar una capa vecina, o de qué nombre derivar el del resultado.
        """
        return self.active_indices[0] if self.active_indices else -1

    @active_index.setter
    def active_index(self, i: int) -> None:
        self.active_indices = [int(i)] if i is not None and i >= 0 else []

    @property
    def activas(self) -> list[CloudLayer]:
        return [self.layers[i] for i in self.active_indices
                if 0 <= i < len(self.layers)]

    def set_activas(self, indices) -> None:
        """Fija el conjunto activo, sin repetidos y en el orden recibido."""
        vistos: list[int] = []
        for i in indices:
            i = int(i)
            if 0 <= i < len(self.layers) and i not in vistos:
                vistos.append(i)
        self.active_indices = vistos

    @property
    def active(self) -> CloudLayer:
        """La capa activa; si hay varias, una capa virtual con la unión.

        Las herramientas solo leen `active.pcd.points`, así que devolver una capa
        combinada las hace funcionar sobre el conjunto sin tocarlas. No pertenece
        al stack y modificarla no modifica nada: para escribir hay que repartir
        con `split_activas` u `origen_activo`.
        """
        caps = self.activas
        if not caps:
            # Sin esto devolvía `layers[-1]`: el conjunto vacío se traducía en
            # silencio a "la última capa", y una herramienta habría operado
            # sobre una capa que nadie eligió.
            raise ValueError("No hay ninguna capa en uso.")
        if len(caps) == 1:
            return caps[0]
        clave = tuple((id(c.pcd), len(c.pcd.points)) for c in caps)
        if self._cache_combinada is not None and self._cache_combinada[0] == clave:
            return self._cache_combinada[1]
        pts = np.vstack([np.asarray(c.pcd.points) for c in caps])
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        if all(len(c.pcd.colors) == len(c.pcd.points) for c in caps):
            pcd.colors = o3d.utility.Vector3dVector(
                np.vstack([np.asarray(c.pcd.colors) for c in caps]))
        capa = CloudLayer(name=f"{len(caps)} capas activas", pcd=pcd)
        self._cache_combinada = (clave, capa)
        return capa

    def origen_activo(self) -> np.ndarray:
        """Índice de capa de cada punto activo, en el orden de `active`."""
        if not self.active_indices:
            return np.empty(0, dtype=np.int32)
        return np.concatenate([
            np.full(len(self.layers[i].pcd.points), i, dtype=np.int32)
            for i in self.active_indices if 0 <= i < len(self.layers)
        ]) if self.activas else np.empty(0, dtype=np.int32)

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

    def extraer_de_activas(self, grupos) -> list[CloudLayer]:
        """Saca subconjuntos de los puntos activos como capas nuevas.

        Cada grupo es `(nombre, máscara sobre los puntos activos)`. Es la
        operación de DBSCAN: los clusters elegidos salen como capas propias y
        sus puntos se quitan de las capas de origen. Una capa que quede vacía
        desaparece. Devuelve las capas creadas, que quedan activas.
        """
        caps = self.activas
        if not caps:
            raise ValueError("No hay ninguna capa activa.")
        origen = self.origen_activo()
        combinada = self.active.pcd

        usados = np.zeros(len(origen), dtype=bool)
        nuevas: list[CloudLayer] = []
        for nombre, m in grupos:
            m = np.asarray(m, dtype=bool)
            if m.any():
                nuevas.append(CloudLayer(name=nombre, pcd=_subset(combinada, m)))
                usados |= m
        if not nuevas:
            raise ValueError("Ninguno de los grupos tiene puntos.")

        # El punto de inserción se calcula antes de tocar nada: después de
        # vaciar capas los índices ya no valen.
        primera = self.active_indices[0]
        vacias = set()
        for i in self.active_indices:
            keep = ~usados[origen == i]
            if keep.any():
                self.layers[i].pcd = _subset(self.layers[i].pcd, keep)
            else:
                vacias.add(i)

        restantes = [c for i, c in enumerate(self.layers) if i not in vacias]
        destino = sum(1 for i in range(primera + 1) if i not in vacias)
        self.layers = restantes[:destino] + nuevas + restantes[destino:]
        self.active_indices = list(range(destino, destino + len(nuevas)))
        self._cache_combinada = None
        return nuevas

    def eliminar_de_activas(self, mask) -> int:
        """Borra los puntos marcados de las capas activas. Devuelve cuántos.

        Es destructivo a propósito: se usa para quitar ruido, y conservar el
        ruido como capa oculta no aporta nada.
        """
        mask = np.asarray(mask, dtype=bool)
        if not self.activas:
            raise ValueError("No hay ninguna capa activa.")
        origen = self.origen_activo()
        if len(mask) != len(origen):
            raise ValueError(
                f"Máscara de largo {len(mask)} para {len(origen)} puntos activos.")
        if mask.all():
            raise ValueError("Eso eliminaría todas las capas activas.")

        for i in self.active_indices:
            keep = ~mask[origen == i]
            if not keep.all():
                self.layers[i].pcd = _subset(self.layers[i].pcd, keep)
        self._cache_combinada = None
        return int(mask.sum())

    def _partir_capa(self, capa, mask, k, sufijos, fine_keep_fn, edits_dir):
        """Parte una capa en dos mitades que heredan su nombre.

        Heredar el nombre es lo que permite seguir sabiendo de qué escaneo viene
        cada punto después de varios cortes.
        """
        dentro = CloudLayer(name=f"{capa.name} · {sufijos[0]} {k}",
                            pcd=_subset(capa.pcd, mask),
                            n_pts_archivo=capa.n_pts_archivo)
        fuera = CloudLayer(name=f"{capa.name} · {sufijos[1]} {k}",
                           pcd=_subset(capa.pcd, ~mask), visible=False,
                           descarte=True, n_pts_archivo=capa.n_pts_archivo)
        if (fine_keep_fn is not None and capa.fine_path is not None
                and Path(capa.fine_path).exists()):
            fino = o3d.io.read_point_cloud(str(capa.fine_path))
            keep_f = fine_keep_fn(fino)
            if keep_f is not None:
                keep_f = np.asarray(keep_f, dtype=bool)
                edits_dir = Path(edits_dir)
                edits_dir.mkdir(parents=True, exist_ok=True)
                base = _slug(capa.name)
                p_in = edits_dir / f"{base}_{sufijos[0]}_{k}.ply"
                p_out = edits_dir / f"{base}_{sufijos[1]}_{k}.ply"
                o3d.io.write_point_cloud(str(p_in), _subset(fino, keep_f))
                o3d.io.write_point_cloud(str(p_out), _subset(fino, ~keep_f))
                dentro.fine_path, fuera.fine_path = p_in, p_out
        return dentro, fuera

    def split_activas(self, keep_mask, fine_keep_fn=None, edits_dir=None,
                      sufijos=("dentro", "fuera")):
        """Aplica una máscara sobre los puntos activos y la reparte por capa.

        `keep_mask` va sobre `active.pcd.points`, o sea sobre la unión de las
        capas activas en el orden de `active_indices`. Cada capa que quede
        partida se reemplaza por sus dos mitades; las que quedan enteras de un
        lado no se dividen, para no llenar la lista de capas vacías.

        A diferencia de `split_active`, no conserva una copia de la capa de
        origen: con varias capas activas eso multiplicaría la memoria, y no se
        pierde nada porque la unión de las dos mitades es exactamente la capa
        original, y la mitad descartada queda marcada y es restaurable.

        Devuelve (n_divididas, n_completas, n_sin_captura).
        """
        keep_mask = np.asarray(keep_mask, dtype=bool)
        if not self.activas:
            raise ValueError("No hay ninguna capa activa.")
        origen = self.origen_activo()
        if len(keep_mask) != len(origen):
            raise ValueError(
                f"Máscara de largo {len(keep_mask)} para {len(origen)} "
                "puntos activos.")
        if not keep_mask.any():
            raise ValueError("La máscara no conserva nada: la capa quedaría vacía.")
        if keep_mask.all():
            raise ValueError("La máscara conserva todo: no hay nada que separar.")

        masks = {i: keep_mask[origen == i] for i in self.active_indices}
        self._split_counter += 1
        k = self._split_counter

        nuevas: list[CloudLayer] = []
        activas: list[int] = []
        n_div = n_comp = n_sin = 0
        for i, capa in enumerate(self.layers):
            mask = masks.get(i)
            if mask is None:                    # capa no activa: no se toca
                nuevas.append(capa)
                continue
            if mask.all():                      # entera capturada
                activas.append(len(nuevas))
                nuevas.append(capa)
                n_comp += 1
                continue
            if not mask.any():                  # no aportó nada
                capa.visible = False
                capa.descarte = True
                nuevas.append(capa)
                n_sin += 1
                continue
            dentro, fuera = self._partir_capa(
                capa, mask, k, sufijos, fine_keep_fn, edits_dir)
            activas.append(len(nuevas))
            nuevas.extend((dentro, fuera))
            n_div += 1

        self.layers = nuevas
        self.active_indices = activas or [0]
        self._cache_combinada = None
        return n_div, n_comp, n_sin

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
        quedan = [j - 1 if j > i else j for j in self.active_indices if j != i]
        self.active_indices = [j for j in quedan if 0 <= j < len(self.layers)] or [0]
        self._cache_combinada = None

    def insert_layer(self, i: int, layer: CloudLayer) -> None:
        """Reinserta una capa (p. ej. al deshacer una eliminación)."""
        i = max(0, min(i, len(self.layers)))
        self.layers.insert(i, layer)
        self.active_indices = [j + 1 if j >= i else j for j in self.active_indices]
        self._cache_combinada = None

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
