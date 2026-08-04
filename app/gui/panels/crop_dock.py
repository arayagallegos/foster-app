"""
crop_dock.py — Dock lateral de recorte: herramienta activa (caja o lazo),
lista de capas y exportación.

No accede a Project ni Viewer: emite señales que MainWindow conecta.
Colores de previsualización en toda la app: VERDE = se conserva, ROJO = se elimina.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup, QDockWidget, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QRadioButton, QSlider,
    QStackedWidget, QVBoxLayout, QWidget,
)

# (etiqueta, valor de la senal, explicacion para tooltip y dialogo de ayuda)
LASSO_OPS = [
    ("Conservar lo del lazo", "union",
     "Lo de dentro del lazo queda VERDE (se conserva). "
     "Varios lazos seguidos van sumando zonas verdes."),
    ("Quitar lo del lazo", "difference",
     "Lo de dentro del lazo queda ROJO (se elimina) y el resto de la capa "
     "queda verde. Ideal para borrar puntos sueltos."),
]

_AYUDA = (
    "<b>Colores</b><br>"
    "VERDE = se conserva &nbsp;·&nbsp; ROJO = se elimina.<br><br>"
    "<b>Lazo</b><br>"
    "1. Elige la operacion.<br>"
    "2. 'Iniciar lazo' y clic sobre la vista para marcar vertices.<br>"
    "3. Cierra con clic derecho, doble clic o clic sobre el primer vertice.<br>"
    "4. Repite si lo necesitas y luego 'Aplicar recorte'.<br><br>"
    "<b>Operaciones del lazo</b><br>"
    + "<br>".join(f"· <b>{lab}</b>: {ayuda}" for lab, _v, ayuda in LASSO_OPS)
    + "<br><br><b>Al aplicar un recorte</b><br>"
    "La capa fuente queda intacta y se oculta; aparecen dos capas nuevas: "
    "'Recorte N' (lo verde, queda activa) y 'Descarte N' (lo rojo, oculta). "
    "Nada se pierde hasta que elimines una capa.<br><br>"
    "<b>Capas</b><br>"
    "Las herramientas operan sobre la capa activa (fila seleccionada). El checkbox "
    "muestra/oculta cada capa. 'Exportar visibles' une las capas visibles en un "
    ".ply — ojo: si dejas visibles la fuente Y su recorte, exportas puntos "
    "duplicados.<br><br>"
    "<b>Camara</b><br>Shift+arrastrar desplaza el encuadre (pan)."
)


class CropDock(QDockWidget):
    """Dock 'Recorte': herramienta activa + capas + exportación."""

    # Herramienta caja
    tool_apply = pyqtSignal()
    tool_cancel = pyqtSignal()
    # Herramienta lazo
    # (tolerancia, phi_min, phi_max)
    esfera_params_changed = pyqtSignal(float, float, float)
    esfera_capturar = pyqtSignal()

    lasso_started = pyqtSignal(str)   # set_op
    lasso_apply = pyqtSignal()
    lasso_cancel = pyqtSignal()
    # Capas
    layer_visibility_changed = pyqtSignal(int, bool)
    layer_activated = pyqtSignal(int)
    layer_removed = pyqtSignal(int)
    layer_restore_requested = pyqtSignal()
    export_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Recorte", parent)
        self.setObjectName("crop_dock")
        self._updating_layers = False
        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI                                                                 #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        contenido = QWidget()
        layout = QVBoxLayout(contenido)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Encabezado: solo el boton de ayuda (toda la explicacion vive ahi)
        fila_top = QWidget()
        ft = QHBoxLayout(fila_top)
        ft.setContentsMargins(0, 0, 0, 0)
        btn_ayuda = QPushButton("ⓘ Ayuda")
        btn_ayuda.setFixedWidth(80)
        btn_ayuda.clicked.connect(self._show_help)
        ft.addStretch(1)
        ft.addWidget(btn_ayuda)
        layout.addWidget(fila_top)

        # ---------- Herramienta activa (pila caja / lazo) ---------- #
        self._tool_box = QGroupBox("Herramienta")
        tool_layout = QVBoxLayout(self._tool_box)
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_page_caja())   # index 0
        self._stack.addWidget(self._build_page_lazo())   # index 1
        self._stack.addWidget(self._build_page_esfera())  # index 2
        tool_layout.addWidget(self._stack)
        layout.addWidget(self._tool_box)

        # ---------- Capas ---------- #
        box_capas = QGroupBox("Capas")
        capas_layout = QVBoxLayout(box_capas)
        self._layer_list = QListWidget()
        self._layer_list.setToolTip(
            "Fila seleccionada = capa activa (donde operan las herramientas).\n"
            "Checkbox = mostrar/ocultar la capa."
        )
        self._layer_list.itemChanged.connect(self._on_item_changed)
        self._layer_list.currentRowChanged.connect(self._on_row_changed)
        capas_layout.addWidget(self._layer_list)

        fila_capas = QWidget()
        fc = QHBoxLayout(fila_capas)
        fc.setContentsMargins(0, 0, 0, 0)
        self._btn_remove_layer = QPushButton("Eliminar")
        self._btn_remove_layer.setToolTip("Elimina la capa seleccionada (pide confirmación).")
        self._btn_remove_layer.setEnabled(False)
        self._btn_remove_layer.clicked.connect(self._emit_layer_removed)
        self._btn_restore_layer = QPushButton("Restaurar eliminada")
        self._btn_restore_layer.setToolTip("Deshace la última eliminación de capa.")
        self._btn_restore_layer.setEnabled(False)
        self._btn_restore_layer.clicked.connect(self.layer_restore_requested)
        fc.addWidget(self._btn_remove_layer)
        fc.addWidget(self._btn_restore_layer)
        capas_layout.addWidget(fila_capas)
        layout.addWidget(box_capas)

        # ---------- Exportar ---------- #
        self._btn_export = QPushButton("Exportar visibles (.ply)")
        self._btn_export.setToolTip("Une las capas visibles y las guarda en un .ply.")
        self._btn_export.clicked.connect(self.export_requested)
        layout.addWidget(self._btn_export)

        layout.addStretch(1)
        self.setWidget(contenido)

    def _build_page_caja(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        instr = QLabel("Arrastra las esferas de la caja en el viewer.")
        instr.setWordWrap(True)
        instr.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        lay.addWidget(instr)
        fila = QWidget()
        fl = QHBoxLayout(fila)
        fl.setContentsMargins(0, 0, 0, 0)
        self._btn_caja_apply = QPushButton("Aplicar recorte")
        self._btn_caja_apply.setToolTip("Conserva lo verde; lo rojo va a una capa Descarte.")
        self._btn_caja_cancel = QPushButton("Cancelar")
        self._btn_caja_apply.clicked.connect(self.tool_apply)
        self._btn_caja_cancel.clicked.connect(self.tool_cancel)
        fl.addWidget(self._btn_caja_apply)
        fl.addWidget(self._btn_caja_cancel)
        lay.addWidget(fila)
        return page

    def _build_page_esfera(self) -> QWidget:
        """Controles de la primitiva esfera: tolerancia y recorte angular.

        Los dos son necesarios y hacen cosas distintas: la TOLERANCIA define
        cuán cerca de la cáscara debe estar un punto; el RECORTE ANGULAR, en qué
        tramo de la esfera se busca. Sin el recorte no existe una tolerancia que
        capture zonas que se desvían (el faldón de la cúpula) sin absorber
        elementos vecinos que están sobre la misma esfera (las compuertas).
        """
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)

        lay.addWidget(QLabel("Arrastra la esfera sobre la nube.\n"
                             "Al soltar se ajusta y muestra lo capturado."))

        self._lbl_tol = QLabel()
        self._sld_tol = QSlider(Qt.Orientation.Horizontal)
        self._sld_tol.setRange(2, 40)            # 0.02 .. 0.40 m
        self._sld_tol.setValue(8)
        self._sld_tol.valueChanged.connect(self._emit_esfera_params)
        lay.addWidget(self._lbl_tol)
        lay.addWidget(self._sld_tol)

        self._lbl_phi = QLabel()
        self._sld_phi_min = QSlider(Qt.Orientation.Horizontal)
        self._sld_phi_min.setRange(0, 180)
        self._sld_phi_min.setValue(0)
        self._sld_phi_max = QSlider(Qt.Orientation.Horizontal)
        self._sld_phi_max.setRange(0, 180)
        self._sld_phi_max.setValue(180)
        for s in (self._sld_phi_min, self._sld_phi_max):
            s.valueChanged.connect(self._emit_esfera_params)
        lay.addWidget(self._lbl_phi)
        lay.addWidget(self._sld_phi_min)
        lay.addWidget(self._sld_phi_max)

        self._btn_capturar = QPushButton("Capturar entidad")
        self._btn_capturar.setToolTip("Lo verde pasa a ser una capa nueva.")
        self._btn_capturar.setEnabled(False)
        self._btn_capturar.clicked.connect(self.esfera_capturar)
        lay.addWidget(self._btn_capturar)

        self._actualizar_labels_esfera()
        return page

    def _actualizar_labels_esfera(self) -> None:
        self._lbl_tol.setText(f"Tolerancia: {self._sld_tol.value() / 100:.2f} m")
        self._lbl_phi.setText(f"Recorte angular φ: "
                              f"{self._sld_phi_min.value()}° – "
                              f"{self._sld_phi_max.value()}°")

    def _emit_esfera_params(self) -> None:
        self._actualizar_labels_esfera()
        self.esfera_params_changed.emit(
            self._sld_tol.value() / 100.0,
            float(self._sld_phi_min.value()),
            float(self._sld_phi_max.value()),
        )

    def set_esfera_has_selection(self, has: bool) -> None:
        self._btn_capturar.setEnabled(bool(has))

    def _build_page_lazo(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)

        self._lasso_op_group = QButtonGroup(self)
        for label, value, ayuda in LASSO_OPS:
            rb = QRadioButton(label)
            rb.setProperty("lasso_op", value)
            rb.setToolTip(ayuda)
            lay.addWidget(rb)
            self._lasso_op_group.addButton(rb)
        self._lasso_op_group.buttons()[0].setChecked(True)

        self._btn_start_lasso = QPushButton("Iniciar lazo")
        self._btn_start_lasso.setToolTip(
            "Clic para marcar vertices; cierra con clic derecho, doble clic "
            "o clic sobre el primer vertice."
        )
        self._btn_apply_lasso = QPushButton("Aplicar recorte")
        self._btn_apply_lasso.setToolTip("Conserva lo verde; lo rojo va a una capa Descarte.")
        self._btn_cancel_lasso = QPushButton("Cancelar lazo")
        self._btn_apply_lasso.setEnabled(False)
        self._btn_cancel_lasso.setEnabled(False)
        self._btn_start_lasso.clicked.connect(self._emit_lasso_started)
        self._btn_apply_lasso.clicked.connect(self.lasso_apply)
        self._btn_cancel_lasso.clicked.connect(self.lasso_cancel)
        lay.addWidget(self._btn_start_lasso)
        lay.addWidget(self._btn_apply_lasso)
        lay.addWidget(self._btn_cancel_lasso)
        return page

    # ------------------------------------------------------------------ #
    # API para MainWindow                                                   #
    # ------------------------------------------------------------------ #

    def set_tool(self, tool: str) -> None:
        """Cambia la página de herramienta: 'caja' o 'lazo'."""
        if tool not in ("caja", "lazo", "esfera"):
            raise ValueError(f"Herramienta desconocida: {tool}")
        self._stack.setCurrentIndex({"caja": 0, "lazo": 1, "esfera": 2}[tool])
        self._tool_box.setTitle(f"Herramienta: {tool.capitalize()}")

    def set_lasso_active(self, active: bool) -> None:
        self._btn_start_lasso.setEnabled(not bool(active))
        self._btn_cancel_lasso.setEnabled(bool(active))

    def set_lasso_has_selection(self, has: bool) -> None:
        self._btn_apply_lasso.setEnabled(bool(has))

    def set_restore_available(self, available: bool) -> None:
        self._btn_restore_layer.setEnabled(bool(available))

    def refresh_layers(self, stack) -> None:
        """Repuebla la lista desde el LayerStack (checkbox=visible, fila=activa)."""
        self._updating_layers = True
        try:
            self._layer_list.clear()
            for capa in stack:
                n = len(capa.pcd.points)
                item = QListWidgetItem(f"{capa.name} ({n:,} pts)")
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked if capa.visible else Qt.CheckState.Unchecked
                )
                self._layer_list.addItem(item)
            self._layer_list.setCurrentRow(stack.active_index)
            self._btn_remove_layer.setEnabled(len(stack) > 1)
        finally:
            self._updating_layers = False

    # ------------------------------------------------------------------ #
    # Internos                                                              #
    # ------------------------------------------------------------------ #

    def _show_help(self) -> None:
        QMessageBox.information(self, "Ayuda — Recorte", _AYUDA)

    def _on_item_changed(self, item) -> None:
        if self._updating_layers:
            return
        i = self._layer_list.row(item)
        self.layer_visibility_changed.emit(
            i, item.checkState() == Qt.CheckState.Checked
        )

    def _on_row_changed(self, row: int) -> None:
        if self._updating_layers or row < 0:
            return
        self.layer_activated.emit(row)

    def _emit_layer_removed(self) -> None:
        row = self._layer_list.currentRow()
        if row >= 0:
            self.layer_removed.emit(row)

    def _emit_lasso_started(self) -> None:
        checked = self._lasso_op_group.checkedButton()
        op = checked.property("lasso_op") if checked else "union"
        self.lasso_started.emit(op)
