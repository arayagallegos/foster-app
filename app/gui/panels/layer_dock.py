"""
layer_dock.py — Dock independiente con la lista de capas.

Se separó del dock de herramientas por dos motivos. Uno de espacio: compartiendo
panel, la lista quedaba al final, bajo los controles de la herramienta activa, y
con 35 capas-scan obligaba a desplazarse sin poder ampliarla. Otro de ciclo de
vida: las capas existen desde que se carga la nube, mientras que la herramienta
solo tiene sentido mientras se usa una.

No accede a Project ni Viewer: emite señales que MainWindow conecta.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDockWidget, QHBoxLayout, QListWidget, QListWidgetItem, QPushButton,
    QStyledItemDelegate, QVBoxLayout, QWidget,
)


class _DelegadoNombreCapa(QStyledItemDelegate):
    """Al editar, muestra SOLO el nombre de la capa.

    La fila se ve como '[descarte] Scan 03 · dentro 1 (12.345 pts)', pero el
    contador y la marca son decoración calculada. Si el usuario editara ese
    texto habría que reconstruir el nombre a partir de él, y bastaría un
    paréntesis escrito a mano para romperlo. El nombre real viaja aparte, en
    UserRole.
    """

    def setEditorData(self, editor, index) -> None:
        editor.setText(index.data(Qt.ItemDataRole.UserRole) or "")

    def setModelData(self, editor, model, index) -> None:
        nombre = editor.text().strip()
        if nombre:      # un nombre vacío dejaría la capa sin identificar
            model.setData(index, nombre, Qt.ItemDataRole.UserRole)


class LayerDock(QDockWidget):
    """Dock 'Capas': lista, visibilidad, entidades y exportación."""

    layer_visibility_changed = pyqtSignal(int, bool)
    layer_activated = pyqtSignal(int)
    layer_removed = pyqtSignal(int)
    layer_renamed = pyqtSignal(int, str)
    layer_restore_requested = pyqtSignal()
    layers_merge_requested = pyqtSignal(list)            # índices a unir
    layers_visibility_changed = pyqtSignal(list, bool)   # índices, visible
    layers_isolate_requested = pyqtSignal(list)          # dejar visibles solo estas
    layers_invert_requested = pyqtSignal()
    discards_removal_requested = pyqtSignal()
    export_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Capas", parent)
        self.setObjectName("layer_dock")
        self._updating = False
        self._nombres: list[str] = []
        self._visibles: list[bool] = []
        self._build_ui()

    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        contenido = QWidget()
        layout = QVBoxLayout(contenido)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Atajos de visibilidad. Con 35 capas-scan, aislar un grupo a mano
        # exige decenas de clics; estos cuatro botones lo resuelven en uno.
        fila_vis = QWidget()
        fv = QHBoxLayout(fila_vis)
        fv.setContentsMargins(0, 0, 0, 0)
        fv.setSpacing(4)
        for etiqueta, ayuda, ranura in (
            ("Solo esta", "Deja visibles únicamente las capas seleccionadas.",
             self._emit_isolate),
            ("Todas", "Muestra todas las capas.",
             lambda: self._emit_visibilidad_todas(True)),
            ("Ninguna", "Oculta todas las capas.",
             lambda: self._emit_visibilidad_todas(False)),
            ("Invertir", "Muestra lo oculto y oculta lo visible.",
             self.layers_invert_requested.emit),
        ):
            b = QPushButton(etiqueta)
            b.setToolTip(ayuda)
            b.clicked.connect(ranura)
            fv.addWidget(b)
        layout.addWidget(fila_vis)

        self._lista = QListWidget()
        self._lista.setToolTip(
            "Fila seleccionada = capa activa (donde operan las herramientas).\n"
            "Checkbox = mostrar/ocultar; con varias seleccionadas, aplica a todas.\n"
            "Doble clic sobre el nombre para renombrar la entidad."
        )
        self._lista.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        # Con 35 capas, colapsar a una o dos filas vuelve la lista inútil.
        self._lista.setMinimumHeight(180)
        self._lista.setItemDelegate(_DelegadoNombreCapa(self._lista))
        self._lista.itemChanged.connect(self._on_item_changed)
        self._lista.currentRowChanged.connect(self._on_row_changed)
        self._lista.itemSelectionChanged.connect(self._on_selection_changed)
        # La lista se lleva todo el espacio sobrante: es el contenido principal
        # del dock y con muchas capas necesita crecer.
        layout.addWidget(self._lista, stretch=1)

        fila = QWidget()
        fc = QHBoxLayout(fila)
        fc.setContentsMargins(0, 0, 0, 0)
        self._btn_eliminar = QPushButton("Eliminar")
        self._btn_eliminar.setToolTip("Elimina la capa seleccionada (pide confirmación).")
        self._btn_eliminar.setEnabled(False)
        self._btn_eliminar.clicked.connect(self._emit_removed)
        self._btn_restaurar = QPushButton("Restaurar eliminada")
        self._btn_restaurar.setToolTip("Deshace la última eliminación de capa.")
        self._btn_restaurar.setEnabled(False)
        self._btn_restaurar.clicked.connect(self.layer_restore_requested)
        fc.addWidget(self._btn_eliminar)
        fc.addWidget(self._btn_restaurar)
        layout.addWidget(fila)

        self._btn_descartes = QPushButton("Eliminar descartes")
        self._btn_descartes.setToolTip(
            "Elimina de una vez todas las capas marcadas como descarte "
            "(lo que quedó fuera de los recortes).")
        self._btn_descartes.setEnabled(False)
        self._btn_descartes.clicked.connect(self.discards_removal_requested)
        layout.addWidget(self._btn_descartes)

        self._btn_unir = QPushButton("Unir seleccionadas")
        self._btn_unir.setToolTip(
            "Funde en una sola capa las capas marcadas (Ctrl o Shift para "
            "marcar varias). Útil cuando una entidad quedó partida.")
        self._btn_unir.setEnabled(False)
        self._btn_unir.clicked.connect(self._emit_merge)
        layout.addWidget(self._btn_unir)

        self._btn_export = QPushButton("Exportar visibles (.ply)")
        self._btn_export.setToolTip("Une las capas visibles y las guarda en un .ply.")
        self._btn_export.clicked.connect(self.export_requested)
        layout.addWidget(self._btn_export)

        self.setWidget(contenido)

    # ------------------------------------------------------------------ #
    # API para MainWindow                                                #
    # ------------------------------------------------------------------ #

    def refresh_layers(self, stack) -> None:
        """Repuebla la lista desde el LayerStack."""
        self._updating = True
        try:
            self._lista.clear()
            for capa in stack:
                n = len(capa.pcd.points)
                descarte = getattr(capa, "descarte", False)
                item = QListWidgetItem(
                    f"{'[descarte] ' if descarte else ''}{capa.name} ({n:,} pts)")
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable
                              | Qt.ItemFlag.ItemIsEditable)
                item.setCheckState(Qt.CheckState.Checked if capa.visible
                                   else Qt.CheckState.Unchecked)
                if descarte:
                    item.setForeground(QColor("#8a6060"))
                item.setData(Qt.ItemDataRole.UserRole, capa.name)
                self._lista.addItem(item)
            self._nombres = [c.name for c in stack]
            self._visibles = [bool(c.visible) for c in stack]
            self._lista.setCurrentRow(stack.active_index)
            self._btn_eliminar.setEnabled(len(stack) > 1)

            n_desc = stack.contar_descartes()
            self._btn_descartes.setEnabled(0 < n_desc < len(stack))
            self._btn_descartes.setText(
                f"Eliminar descartes ({n_desc})" if n_desc else "Eliminar descartes")
        finally:
            self._updating = False

    def set_restore_enabled(self, activo: bool) -> None:
        self._btn_restaurar.setEnabled(bool(activo))

    # ------------------------------------------------------------------ #
    # Internos                                                            #
    # ------------------------------------------------------------------ #

    def _on_item_changed(self, item) -> None:
        """itemChanged cubre tanto el checkbox como el renombre: hay que mirar
        qué cambió respecto del último refresco para saber cuál de los dos fue."""
        if self._updating:
            return
        i = self._lista.row(item)
        if not 0 <= i < len(self._nombres):
            return
        nombre = item.data(Qt.ItemDataRole.UserRole)
        if nombre and nombre != self._nombres[i]:
            self._nombres[i] = nombre
            self.layer_renamed.emit(i, nombre)
            return

        visible = item.checkState() == Qt.CheckState.Checked
        self._visibles[i] = visible

        # Si la capa tocada forma parte de una selección múltiple, el cambio se
        # aplica a todas: es lo que hace cualquier panel de capas.
        filas = [self._lista.row(it) for it in self._lista.selectedItems()]
        if len(filas) > 1 and i in filas:
            for f in filas:
                self._visibles[f] = visible
            self.layers_visibility_changed.emit(sorted(filas), visible)
            return
        self.layer_visibility_changed.emit(i, visible)

    def _on_row_changed(self, fila: int) -> None:
        if not self._updating and fila >= 0:
            self.layer_activated.emit(fila)

    def _on_selection_changed(self) -> None:
        self._btn_unir.setEnabled(len(self._lista.selectedItems()) >= 2)

    def _filas_seleccionadas(self) -> list[int]:
        return sorted(self._lista.row(it) for it in self._lista.selectedItems())

    def _emit_isolate(self) -> None:
        if filas := self._filas_seleccionadas():
            self.layers_isolate_requested.emit(filas)

    def _emit_visibilidad_todas(self, visible: bool) -> None:
        self.layers_visibility_changed.emit(list(range(self._lista.count())), visible)

    def _emit_merge(self) -> None:
        filas = self._filas_seleccionadas()
        if len(filas) >= 2:
            self.layers_merge_requested.emit(filas)

    def _emit_removed(self) -> None:
        fila = self._lista.currentRow()
        if fila >= 0:
            self.layer_removed.emit(fila)
