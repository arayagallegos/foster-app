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


class RegistrationWorker(QThread):
    """
    Registra múltiples scans de un e57 en un hilo separado.
    Emite progreso durante el proceso (puede tardar varios minutos).
    """

    finished = pyqtSignal(object, str)  # (PointCloud registrada, path)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int)
    status   = pyqtSignal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self._path = path

    def run(self):
        try:
            from app.core.io import load_e57_scans
            from app.modules.fusion import register_scans

            self.status.emit("Leyendo scans individuales...")
            self.progress.emit(5)

            scans = load_e57_scans(self._path, voxel_size=0.05)

            if not scans:
                self.error.emit("No se encontraron scans válidos en el archivo.")
                return

            if len(scans) == 1:
                # Solo un scan: no hay nada que registrar
                self.status.emit("Solo un scan encontrado, sin necesidad de registro.")
                self.progress.emit(100)
                self.finished.emit(scans[0][0], self._path)
                return

            def on_progress(pct, msg):
                self.progress.emit(5 + int(pct * 0.9))
                self.status.emit(msg)

            registered = register_scans(scans, progress_cb=on_progress)
            self.progress.emit(100)
            self.finished.emit(registered, self._path)

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
