"""
crop_panel.py — Panel flotante para el modo de recorte interactivo.

No accede directamente al Project ni al Viewer.
Emite senales Qt que el MainWindow conecta a la logica de negocio.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QCheckBox,
    QFrame, QButtonGroup, QRadioButton,
)
from PyQt6.QtCore import pyqtSignal, Qt


class CropPanel(QWidget):
    """
    Panel flotante que aparece al activar el modo recorte.

    Estados:
    - Modo activo: muestra fuente + instruccion + botones Aplicar/Cancelar
    - Post-crop:   ademas muestra checkboxes de visibilidad
    - Lazo:        seccion siempre visible al final con radio buttons y botones
    """

    apply_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    export_requested = pyqtSignal()
    visibility_changed = pyqtSignal(bool, bool)  # (show_original, show_cropped)

    lasso_started = pyqtSignal(str)       # set_op
    lasso_apply_requested = pyqtSignal()
    lasso_cancel_requested = pyqtSignal()

    def __init__(self, source_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui(source_name)

    def _build_ui(self, source_name: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        layout.addWidget(QLabel(f"Fuente: <b>{source_name}</b>"))
        layout.addWidget(QLabel("Ajusta la caja en el viewer."))

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)

        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)

        self._btn_apply = QPushButton("Aplicar")
        self._btn_cancel = QPushButton("Cancelar")
        btn_layout.addWidget(self._btn_apply)
        btn_layout.addWidget(self._btn_cancel)
        layout.addWidget(btn_row)

        self._chk_original = QCheckBox("Nube original")
        self._chk_cropped = QCheckBox("Nube recortada")
        self._chk_original.setChecked(True)
        self._chk_cropped.setChecked(True)
        self._chk_original.hide()
        self._chk_cropped.hide()
        layout.addWidget(self._chk_original)
        layout.addWidget(self._chk_cropped)

        self._btn_export = QPushButton("Exportar recorte (.ply)")
        self._btn_export.hide()
        layout.addWidget(self._btn_export)

        # Lasso section
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep2)

        layout.addWidget(QLabel("<b>Lazo</b>"))

        self._lasso_op_group = QButtonGroup(self)
        lasso_ops = [
            ("Union", "union"),
            ("Interseccion", "intersection"),
            ("Diferencia", "difference"),
            ("Dif. simetrica", "symmetric_difference"),
        ]
        for label, value in lasso_ops:
            rb = QRadioButton(label)
            rb.setProperty("lasso_op", value)
            layout.addWidget(rb)
            self._lasso_op_group.addButton(rb)
        self._lasso_op_group.buttons()[0].setChecked(True)

        self._btn_start_lasso = QPushButton("Iniciar lazo")
        self._btn_apply_lasso = QPushButton("Aplicar seleccion")
        self._btn_cancel_lasso = QPushButton("Cancelar lazo")
        self._btn_apply_lasso.setEnabled(False)
        self._btn_cancel_lasso.setEnabled(False)

        layout.addWidget(self._btn_start_lasso)
        layout.addWidget(self._btn_apply_lasso)
        layout.addWidget(self._btn_cancel_lasso)

        self._btn_apply.clicked.connect(self.apply_requested)
        self._btn_cancel.clicked.connect(self.cancel_requested)
        self._btn_export.clicked.connect(self.export_requested)
        self._chk_original.toggled.connect(self._emit_visibility)
        self._chk_cropped.toggled.connect(self._emit_visibility)

        self._btn_start_lasso.clicked.connect(self._emit_lasso_started)
        self._btn_apply_lasso.clicked.connect(self.lasso_apply_requested)
        self._btn_cancel_lasso.clicked.connect(self.lasso_cancel_requested)

        self.adjustSize()

    def show_visibility_controls(self) -> None:
        """Muestra los toggles de visibilidad y exportacion tras un crop exitoso."""
        self._btn_apply.setText("Re-aplicar")
        self._chk_original.show()
        self._chk_cropped.show()
        self._btn_export.show()
        self.adjustSize()

    def set_lasso_active(self, active: bool) -> None:
        self._btn_start_lasso.setEnabled(not active)
        self._btn_cancel_lasso.setEnabled(active)

    def set_lasso_has_selection(self, has: bool) -> None:
        self._btn_apply_lasso.setEnabled(has)

    def _emit_lasso_started(self) -> None:
        checked = self._lasso_op_group.checkedButton()
        op = checked.property("lasso_op") if checked else "union"
        self.lasso_started.emit(op)

    def _emit_visibility(self) -> None:
        self.visibility_changed.emit(
            self._chk_original.isChecked(),
            self._chk_cropped.isChecked(),
        )
