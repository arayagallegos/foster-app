"""
crop_panel.py — Panel flotante para el modo de recorte interactivo.

No accede directamente al Project ni al Viewer.
Emite senales Qt que el MainWindow conecta a la logica de negocio.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLayout,
    QPushButton, QLabel, QCheckBox,
    QGroupBox, QButtonGroup, QRadioButton,
)
from PyQt6.QtCore import pyqtSignal, Qt


# (etiqueta, valor de la senal, explicacion mostrada bajo los radios)
LASSO_OPS = [
    ("Agregar (union)", "union",
     "Los puntos dentro del lazo se SUMAN a la seleccion amarilla."),
    ("Intersectar", "intersection",
     "Queda seleccionado solo lo que YA estaba amarillo y ademas cae dentro del lazo. "
     "Sin seleccion previa, parte de toda la nube."),
    ("Quitar (diferencia)", "difference",
     "Los puntos dentro del lazo se QUITAN de la seleccion amarilla. "
     "Sin seleccion previa, parte de toda la nube (util para borrar puntos sueltos)."),
    ("Invertir (dif. simetrica)", "symmetric_difference",
     "Dentro del lazo se invierte: lo amarillo se quita y lo no seleccionado se agrega."),
]

_AYUDA_LAZO = (
    "El lazo arma una seleccion (en amarillo) dibujando poligonos sobre la vista:\n"
    "1. Elige como combinar el proximo lazo con la seleccion.\n"
    "2. Pulsa 'Iniciar lazo' y haz clic para marcar vertices.\n"
    "3. Cierra con clic derecho, doble clic o clic sobre el primer vertice.\n"
    "4. Repite con otra operacion si lo necesitas.\n"
    "5. 'Aplicar seleccion' conserva SOLO lo amarillo."
)

_ANCHO_TEXTO = 250  # px; fija el ancho de los textos para que el wrap sea estable


class CropPanel(QWidget):
    """
    Panel flotante que aparece al activar el modo recorte.

    Estados:
    - Modo activo: muestra fuente + instruccion + botones Aplicar/Cancelar
    - Post-crop:   ademas muestra checkboxes de visibilidad
    - Lazo:        seccion siempre visible al final
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

    # ------------------------------------------------------------------ #
    # Construccion de la UI                                                #
    # ------------------------------------------------------------------ #

    def _texto(self, texto: str, atenuado: bool = False) -> QLabel:
        lbl = QLabel(texto)
        lbl.setWordWrap(True)
        lbl.setFixedWidth(_ANCHO_TEXTO)
        if atenuado:
            lbl.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        return lbl

    def _build_ui(self, source_name: str) -> None:
        # Fondo opaco: sin esto el panel es transparente y deja ver widgets
        # o instancias viejas que queden detras.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            "CropPanel { background-color: #1a1a1a; border: 1px solid #444444; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        # El panel siempre calza con su contenido: evita solapamientos cuando
        # se muestran/ocultan controles (checkboxes post-recorte, etc.)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

        layout.addWidget(self._texto(f"Fuente: <b>{source_name}</b>"))

        # ---------- Seccion 1: recorte por caja ---------- #
        box_caja = QGroupBox("1 · Recorte por caja")
        caja_layout = QVBoxLayout(box_caja)
        caja_layout.setSpacing(6)

        caja_layout.addWidget(self._texto(
            "Arrastra las esferas de la caja en el viewer para ajustarla. "
            "'Aplicar' conserva solo los puntos de adentro.", atenuado=True
        ))

        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        self._btn_apply = QPushButton("Aplicar")
        self._btn_apply.setToolTip("Recorta la nube a la caja actual.")
        self._btn_cancel = QPushButton("Cancelar")
        self._btn_cancel.setToolTip("Sale del modo recorte sin cambios.")
        btn_layout.addWidget(self._btn_apply)
        btn_layout.addWidget(self._btn_cancel)
        caja_layout.addWidget(btn_row)

        self._chk_original = QCheckBox("Mostrar nube original")
        self._chk_original.setToolTip("Muestra/oculta la nube completa de fondo.")
        self._chk_cropped = QCheckBox("Mostrar nube recortada")
        self._chk_cropped.setToolTip("Muestra/oculta el resultado del recorte.")
        self._chk_original.setChecked(True)
        self._chk_cropped.setChecked(True)
        self._chk_original.hide()
        self._chk_cropped.hide()
        caja_layout.addWidget(self._chk_original)
        caja_layout.addWidget(self._chk_cropped)

        self._btn_export = QPushButton("Exportar recorte (.ply)")
        self._btn_export.setToolTip("Guarda la nube recortada en un archivo .ply.")
        self._btn_export.hide()
        caja_layout.addWidget(self._btn_export)

        layout.addWidget(box_caja)

        # ---------- Seccion 2: seleccion por lazo ---------- #
        box_lazo = QGroupBox("2 · Seleccion por lazo")
        lazo_layout = QVBoxLayout(box_lazo)
        lazo_layout.setSpacing(6)

        lazo_layout.addWidget(self._texto(_AYUDA_LAZO, atenuado=True))

        self._lasso_op_group = QButtonGroup(self)
        for label, value, ayuda in LASSO_OPS:
            rb = QRadioButton(label)
            rb.setProperty("lasso_op", value)
            rb.setToolTip(ayuda)
            lazo_layout.addWidget(rb)
            self._lasso_op_group.addButton(rb)
        self._lasso_op_group.buttons()[0].setChecked(True)

        # Explicacion dinamica de la operacion elegida
        self._lasso_hint = self._texto(LASSO_OPS[0][2], atenuado=True)
        lazo_layout.addWidget(self._lasso_hint)
        self._lasso_op_group.buttonToggled.connect(self._update_lasso_hint)

        self._btn_start_lasso = QPushButton("Iniciar lazo")
        self._btn_start_lasso.setToolTip(
            "Entra en modo dibujo: cada clic sobre la vista agrega un vertice."
        )
        self._btn_apply_lasso = QPushButton("Aplicar seleccion")
        self._btn_apply_lasso.setToolTip(
            "Conserva solo los puntos amarillos. Se habilita cuando hay seleccion."
        )
        self._btn_cancel_lasso = QPushButton("Cancelar lazo")
        self._btn_cancel_lasso.setToolTip("Descarta el poligono en curso.")
        self._btn_apply_lasso.setEnabled(False)
        self._btn_cancel_lasso.setEnabled(False)

        lazo_layout.addWidget(self._btn_start_lasso)
        lazo_layout.addWidget(self._btn_apply_lasso)
        lazo_layout.addWidget(self._btn_cancel_lasso)

        layout.addWidget(box_lazo)

        # ---------- Conexiones ---------- #
        self._btn_apply.clicked.connect(self.apply_requested)
        self._btn_cancel.clicked.connect(self.cancel_requested)
        self._btn_export.clicked.connect(self.export_requested)
        self._chk_original.toggled.connect(self._emit_visibility)
        self._chk_cropped.toggled.connect(self._emit_visibility)

        self._btn_start_lasso.clicked.connect(self._emit_lasso_started)
        self._btn_apply_lasso.clicked.connect(self.lasso_apply_requested)
        self._btn_cancel_lasso.clicked.connect(self.lasso_cancel_requested)

    # ------------------------------------------------------------------ #
    # API usada por MainWindow                                             #
    # ------------------------------------------------------------------ #

    def show_visibility_controls(self) -> None:
        """Muestra los toggles de visibilidad y exportacion tras un crop exitoso."""
        self._btn_apply.setText("Re-aplicar")
        self._chk_original.show()
        self._chk_cropped.show()
        self._btn_export.show()
        self.adjustSize()

    def set_lasso_active(self, active: bool) -> None:
        # bool() defensivo: PyQt6 rechaza numpy.bool_ y similares
        self._btn_start_lasso.setEnabled(not bool(active))
        self._btn_cancel_lasso.setEnabled(bool(active))

    def set_lasso_has_selection(self, has: bool) -> None:
        self._btn_apply_lasso.setEnabled(bool(has))

    # ------------------------------------------------------------------ #
    # Internos                                                             #
    # ------------------------------------------------------------------ #

    def _update_lasso_hint(self, boton, checked: bool) -> None:
        if checked:
            op = boton.property("lasso_op")
            for _label, value, ayuda in LASSO_OPS:
                if value == op:
                    self._lasso_hint.setText(ayuda)
                    return

    def _emit_lasso_started(self) -> None:
        checked = self._lasso_op_group.checkedButton()
        op = checked.property("lasso_op") if checked else "union"
        self.lasso_started.emit(op)

    def _emit_visibility(self) -> None:
        self.visibility_changed.emit(
            self._chk_original.isChecked(),
            self._chk_cropped.isChecked(),
        )
