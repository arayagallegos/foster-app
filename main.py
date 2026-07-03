"""
main.py — Punto de entrada de Foster App.

Uso:
    python main.py
"""

import sys
import os

# Necesario en algunos sistemas para que PyVista encuentre Qt
os.environ.setdefault("QT_API", "pyqt6")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
import pyvista as pv


def main():
    # PyVista debe saber que usamos Qt antes de crear la app
    pv.set_plot_theme("dark")

    app = QApplication(sys.argv)
    app.setApplicationName("FosterApp")
    app.setOrganizationName("LabPatrimonio")

    # Importar aquí para que PyQt ya esté inicializado
    from app.gui.main_window import MainWindow

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
