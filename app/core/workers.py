"""
workers.py — QThreads para operaciones que no deben bloquear la UI.

Regla de oro en Qt: cualquier operación que tome más de ~100ms
debe correr en un hilo separado. De lo contrario la ventana se congela.

Uso estándar:
    worker = LoadWorker(path)
    worker.finished.connect(on_cloud_loaded)
    worker.error.connect(on_error)
    worker.start()
"""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal
import open3d as o3d

from app.core.io import load_point_cloud


class LoadWorker(QThread):
    """
    Carga una nube de puntos en un hilo separado.

    Señales:
        finished(pcd, path): emite la nube cargada y la ruta original.
        error(message):      emite un mensaje de error si algo falla.
        progress(int):       progreso 0-100 (por ahora solo 0 y 100).
    """

    finished = pyqtSignal(object, str)  # (PointCloud, path)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self._path = path

    def run(self):
        try:
            self.progress.emit(10)
            pcd = load_point_cloud(self._path)
            self.progress.emit(100)
            self.finished.emit(pcd, self._path)
        except Exception as ex:
            self.error.emit(str(ex))


class ScanLayersWorker(QThread):
    """Carga los scans del .e57 como capas (con caché) en un hilo aparte."""

    finished = pyqtSignal(object)   # list[ScanCacheInfo]
    error    = pyqtSignal(str)
    progress = pyqtSignal(int)
    status   = pyqtSignal(str)

    def __init__(self, path: str, cache_dir, parent=None):
        super().__init__(parent)
        self._path = path
        self._cache_dir = cache_dir

    def run(self):
        try:
            from app.core.io import load_e57_scans_cached

            def cb(pct, msg):
                self.progress.emit(int(pct))
                self.status.emit(msg)

            scans = load_e57_scans_cached(self._path, self._cache_dir, progress_cb=cb)
            if not scans:
                self.error.emit("No se cargó ningún scan válido del archivo.")
                return
            self.finished.emit(scans)
        except Exception as ex:
            self.error.emit(str(ex))


class RecacheWorker(QThread):
    """
    Re-lee el .e57 quedándose solo con lo que cae dentro de una caja, y lo
    cachea con un vóxel más fino. Es la operación más cara de la aplicación
    (hay que releer el archivo completo, porque el caché guarda los puntos ya
    voxelizados y ese detalle no se puede recuperar de otra forma), así que va
    obligatoriamente en un hilo aparte.
    """

    finished = pyqtSignal(object)   # list[ScanCacheInfo]
    error    = pyqtSignal(str)
    progress = pyqtSignal(int)
    status   = pyqtSignal(str)

    def __init__(self, path, cache_dir, min_bound, max_bound,
                 voxel_fino=0.01, voxel_grueso=0.03, parent=None):
        super().__init__(parent)
        self._path = path
        self._cache_dir = cache_dir
        self._min_bound = min_bound
        self._max_bound = max_bound
        self._voxel_fino = voxel_fino
        self._voxel_grueso = voxel_grueso

    def run(self):
        try:
            from app.core.io import recachear_e57_recortado

            def cb(pct, msg):
                self.progress.emit(int(pct))
                self.status.emit(msg)

            scans = recachear_e57_recortado(
                self._path, self._cache_dir, self._min_bound, self._max_bound,
                voxel_fino=self._voxel_fino, voxel_grueso=self._voxel_grueso,
                progress_cb=cb,
            )
            if not scans:
                self.error.emit(
                    "El recorte no dejó puntos en ningún scan. "
                    "Revisa que la caja esté sobre la estructura."
                )
                return
            self.finished.emit(scans)
        except Exception as ex:
            self.error.emit(str(ex))


class DbscanWorker(QThread):
    """Agrupa por densidad en un hilo aparte.

    Aunque `clusterizar` submuestrea, sobre una capa grande sigue costando
    segundos: bloquear la UI dejaría la ventana congelada sin explicación.
    """

    finished = pyqtSignal(object)   # np.ndarray de etiquetas
    error    = pyqtSignal(str)

    def __init__(self, pts, eps: float, min_points: int, parent=None):
        super().__init__(parent)
        self._pts = pts
        self._eps = eps
        self._min_points = min_points

    def run(self):
        try:
            from app.modules.clusters import clusterizar
            self.finished.emit(
                clusterizar(self._pts, eps=self._eps, min_points=self._min_points))
        except Exception as ex:
            self.error.emit(str(ex))


class SimetriaWorker(QThread):
    """Refina el plano de simetría en un hilo aparte.

    Nelder-Mead evalúa el objetivo cientos de veces; sobre una entidad grande
    son decenas de segundos incluso con las consultas paralelizadas.
    """

    finished = pyqtSignal(object, float)   # (PlanoSimetria, concordancia)
    error    = pyqtSignal(str)

    def __init__(self, pts, plano, tolerancia, zona=None, parent=None):
        super().__init__(parent)
        self._pts = pts
        self._plano = plano
        self._tolerancia = tolerancia
        self._zona = zona

    def run(self):
        try:
            from app.modules.simetria import refinar
            plano, conc = refinar(self._pts, self._plano,
                                  tolerancia=self._tolerancia, zona=self._zona)
            self.finished.emit(plano, conc)
        except Exception as ex:
            self.error.emit(str(ex))


class MallaWorker(QThread):
    """Reconstruye la malla de una entidad y mide su fidelidad, en un hilo.

    Es la operación más cara de la herramienta después de la carga: la
    triangulación recorre toda la entidad y la medida de fidelidad muestrea la
    superficie resultante para compararla en las dos direcciones.
    """

    finished = pyqtSignal(object, object, object)   # (Malla, Fidelidad, dict)
    error    = pyqtSignal(str)

    def __init__(self, pts, max_perimetro: float, rellenar: bool,
                 perimetro_relleno: float, parent=None):
        super().__init__(parent)
        self._pts = pts
        self._max_perimetro = max_perimetro
        self._rellenar = rellenar
        self._perimetro_relleno = perimetro_relleno

    def run(self):
        try:
            from app.modules.meshing import (agujeros, diagnostico, fidelidad,
                                             reconstruir, rellenar_agujeros)
            # Se trianguIa sin rellenar para poder contar los contornos antes y
            # después: saber cuántos se cerraron es lo que dice si mover el
            # umbral serviría de algo.
            malla = reconstruir(self._pts, max_perimetro=self._max_perimetro,
                                rellenar=False)
            st = {}
            try:
                antes = agujeros(malla)
                st["n_antes"] = len(antes)
            except Exception:      # el diagnóstico es informativo, no crítico
                antes = None
            if self._rellenar and self._perimetro_relleno > 0:
                malla = rellenar_agujeros(malla, self._perimetro_relleno)
            fid = fidelidad(malla, self._pts)
            try:
                st.update(diagnostico(malla))
                abiertos = agujeros(malla)
                if abiertos:
                    st["mayor"] = abiertos[0]["perimetro"]
                    st["mayor_aristas"] = abiertos[0].get("aristas")
                else:
                    st["mayor"] = 0.0
            except Exception:
                pass
            self.finished.emit(malla, fid, st)
        except Exception as ex:
            self.error.emit(str(ex))


class BaseWorker(QThread):
    """
    Worker genérico para operaciones futuras (ICP, Poisson, etc.).
    Heredar de esta clase para implementar operaciones pesadas.
    """

    finished = pyqtSignal(object)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int)
    status   = pyqtSignal(str)  # mensaje de texto para la barra de estado

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False

    def cancel(self):
        """Solicita cancelación. La subclase debe revisar self._cancelled."""
        self._cancelled = True

    def run(self):
        raise NotImplementedError("Implementar en subclase")
