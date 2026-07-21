"""
conftest.py — Configuración compartida de pytest.

Mitiga un crash de teardown de VTK/OpenGL en Windows: al crear muchas ventanas Qt
(cada una con un contexto OpenGL de VTK) en un mismo proceso de test, los contextos
no se liberan a tiempo y el proceso termina cayéndose. Forzar el procesamiento de
las eliminaciones diferidas de Qt + garbage collection entre tests libera los
contextos y mantiene la suite completa estable.
"""
import gc

import pytest


@pytest.fixture(autouse=True)
def _liberar_contextos_qt():
    yield
    # Cerrar explícitamente todos los plotters de PyVista (liberan su ventana
    # de render VTK); es lo que evita el crash de teardown OpenGL en Windows.
    try:
        import pyvista as pv
        pv.close_all()
    except Exception:
        pass
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is not None:
        app.processEvents()      # ejecuta deleteLater() pendientes
    gc.collect()
    if app is not None:
        app.processEvents()
