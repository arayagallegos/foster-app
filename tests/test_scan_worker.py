"""tests/test_scan_worker.py — Worker de carga de scans como capas."""
import sys

import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(sys.argv)


def test_scan_worker_tiene_senales(app, tmp_path):
    from app.core.workers import ScanLayersWorker
    w = ScanLayersWorker("x.e57", tmp_path)
    for s in ("finished", "error", "progress", "status"):
        assert hasattr(w, s)


def test_scan_worker_emite_error_si_no_existe(app, tmp_path, qtbot):
    from app.core.workers import ScanLayersWorker
    w = ScanLayersWorker(str(tmp_path / "no_existe.e57"), tmp_path / "cache")
    with qtbot.waitSignal(w.error, timeout=5000):
        w.run()   # ejecutar en el hilo del test (no start) para capturar la señal
