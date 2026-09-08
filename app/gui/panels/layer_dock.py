"""
layer_dock.py — Dock independiente con la lista de capas.

Se separó del dock de herramientas por dos motivos. Uno de espacio: compartiendo
panel, la lista quedaba al final, bajo los controles de la herramienta activa, y
con 35 capas-scan obligaba a desplazarse sin poder ampliarla. Otro de ciclo de
vida: las capas existen desde que se carga la nube, mientras que la herramienta
solo tiene sentido mientras se usa una.

La lista es una tabla —Ver, Usar, Capa, En caché, En el archivo— y no una lista
simple. El motivo es que hay DOS condiciones independientes por capa: si se dibuja y si
las herramientas operan sobre ella. Ambas admiten varias capas a la vez, así que
ambas necesitan un control persistente. Antes solo había una casilla y la segunda
condición colgaba de la selección azul, que es transitoria: un clic la destruye.
Eso hacía imposible saber, mirando la lista, sobre qué iba a actuar una
herramienta. Con dos columnas y encabezado, cada condición se lee y se cambia
donde dice lo que es, y el resaltado azul recupera su papel de siempre: elegir
sobre qué filas actúan los botones del panel.

No accede a Project ni Viewer: emite señales que MainWindow conecta.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDockWidget, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QStyledItemDelegate,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

COL_VER, COL_USAR, COL_NOMBRE, COL_PTS, COL_ARCHIVO = 0, 1, 2, 3, 4


class _DelegadoNombreCapa(QStyledItemDelegate):
    """Edita SOLO el nombre, y solo en la columna del nombre.

    La celda se ve como '[descarte] Scan 03 · dentro 1 (12.345 pts)', pero el
    contador y la marca son decoración calculada. Si el usuario editara ese
    texto habría que reconstruir el nombre a partir de él, y bastaría un
    paréntesis escrito a mano para romperlo. El nombre real viaja aparte, en
    UserRole.
    """

    def createEditor(self, parent, option, index):
        if index.column() != COL_NOMBRE:
            return None          # las columnas de casilla no se editan a mano
        return super().createEditor(parent, option, index)

    def setEditorData(self, editor, index) -> None:
        editor.setText(index.data(Qt.ItemDataRole.UserRole) or "")

    def setModelData(self, editor, model, index) -> None:
        nombre = editor.text().strip()
        if nombre:      # un nombre vacío dejaría la capa sin identificar
            model.setData(index, nombre, Qt.ItemDataRole.UserRole)


class LayerDock(QDockWidget):
    """Dock 'Capas': tabla, visibilidad, entidades y exportación."""

    # Control segmentado de dos posiciones para elegir el destino de los atajos.
    # Encendido tiene que distinguirse de un vistazo, porque de él depende qué
    # hacen los cuatro botones de debajo.
    _ESTILO_SEGMENTO = """
        QPushButton {
            background-color: #3c3c3c;
            color: #999999;
            border: 1px solid #555555;
            border-right-width: %(borde_der)s;
            border-top-left-radius: %(radio_izq)s;
            border-bottom-left-radius: %(radio_izq)s;
            border-top-right-radius: %(radio_der)s;
            border-bottom-right-radius: %(radio_der)s;
            padding: 4px 14px;
        }
        QPushButton:hover { background-color: #4a4a4a; }
        QPushButton:checked {
            background-color: #4d7ea8;
            color: #ffffff;
            border-color: #5d8eb8;
            font-weight: 600;
        }
        QPushButton:disabled { color: #666666; }
    """

    layer_visibility_changed = pyqtSignal(int, bool)
    layer_activated = pyqtSignal(int)
    layers_activated = pyqtSignal(list)   # capas marcadas en 'Usar'
    layer_removed = pyqtSignal(int)
    layers_removed = pyqtSignal(list)     # eliminar toda la selección
    layer_renamed = pyqtSignal(int, str)
    layer_restore_requested = pyqtSignal()
    layers_merge_requested = pyqtSignal(list)            # índices a unir
    layers_visibility_changed = pyqtSignal(list, bool)   # índices, visible
    layers_isolate_requested = pyqtSignal(list)          # dejar visibles solo estas
    layers_invert_requested = pyqtSignal()
    activate_visible_requested = pyqtSignal()            # 'Usar' ← 'Ver'
    discards_removal_requested = pyqtSignal()
    export_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Capas", parent)
        self.setObjectName("layer_dock")
        self._updating = False
        self._botones: dict[str, QPushButton] = {}
        self._nombres: list[str] = []
        self._visibles: list[bool] = []
        self._activas: list[bool] = []
        self._build_ui()

    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        contenido = QWidget()
        layout = QVBoxLayout(contenido)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Atajos en bloque. Con 35 capas-scan, montar un grupo a mano exige
        # decenas de clics.
        #
        # Los interruptores eligen a qué columna se aplican, y pueden estar los
        # dos a la vez. Eso evita duplicar los cuatro botones —el panel va justo
        # de espacio— y sobre todo permite el caso más frecuente en un solo
        # clic: aislar unas capas Y ponerlas a trabajar. Con una fila por
        # columna eso eran dos operaciones separadas.
        fila_destino = QWidget()
        fd = QHBoxLayout(fila_destino)
        fd.setContentsMargins(0, 0, 0, 0)
        fd.setSpacing(4)
        fd.addWidget(QLabel("Aplicar a:"))

        # Los dos van pegados, como un interruptor de dos posiciones, para que
        # se lean como una sola pregunta —"¿sobre qué?"— y no como dos botones
        # sueltos. Pero NO son excluyentes: se pueden dejar los dos encendidos,
        # que es el caso que motivó el diseño.
        segmento = QWidget()
        seg = QHBoxLayout(segmento)
        seg.setContentsMargins(0, 0, 0, 0)
        seg.setSpacing(0)           # sin hueco: es lo que los une
        self._destinos: dict[str, QPushButton] = {}
        posiciones = (
            ("Ver", "izq", "Los atajos actuarán sobre la visibilidad."),
            ("Usar", "der", "Los atajos actuarán sobre las capas en uso."),
        )
        for clave, lado, ayuda in posiciones:
            b = QPushButton(clave)
            b.setCheckable(True)
            b.setChecked(True)      # por defecto, ambas van juntas
            b.setToolTip(ayuda)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet(self._ESTILO_SEGMENTO % {
                "radio_izq": "4px" if lado == "izq" else "0",
                "radio_der": "4px" if lado == "der" else "0",
                # El borde compartido se dibuja una sola vez, o se vería doble.
                "borde_der": "0" if lado == "izq" else "1px",
            })
            b.clicked.connect(self._on_destino_cambiado)
            seg.addWidget(b)
            self._destinos[clave] = b
        fd.addWidget(segmento)
        fd.addStretch(1)
        layout.addWidget(fila_destino)

        layout.addWidget(self._fila_atajos("", (
            ("Solo esta", "Deja únicamente las capas seleccionadas.",
             self._atajo_solo_esta),
            ("Todas", "Marca todas las capas.",
             self._atajo_todas),
            ("Ninguna", "Desmarca todas las capas.",
             self._atajo_ninguna),
            ("Invertir", "Cambia cada capa por su contrario.",
             self._atajo_invertir),
        )))
        layout.addWidget(self._fila_atajos("", (
            ("Usar lo que se ve",
             "Pone en uso exactamente las capas que están visibles.",
             self.activate_visible_requested.emit),
        )))
        self._on_destino_cambiado()

        self._arbol = QTreeWidget()
        self._arbol.setColumnCount(5)
        # Los encabezados repiten la fórmula del panel de Información ("Puntos
        # del archivo" / "En el caché"), para que los dos sitios se expliquen
        # mutuamente. Antes iban las dos cifras pegadas dentro del nombre, sin
        # etiqueta que dijera cuál era cuál.
        self._arbol.setHeaderLabels(
            ["Ver", "Usar", "Capa", "En caché", "En el archivo"])
        self._arbol.setRootIsDecorated(False)      # tabla plana, sin flechas
        self._arbol.setToolTip(
            "Ver: si los puntos de la capa se dibujan.\n"
            "Usar: si las herramientas operan sobre ella. Pueden ser varias.\n"
            "En caché: los puntos con los que operan las herramientas.\n"
            "En el archivo: los que traía el escaneo antes de submuestrear.\n"
            "El resaltado azul es solo la selección, para los botones de abajo.\n"
            "Doble clic sobre el nombre para renombrar la entidad."
        )
        cab = self._arbol.header()
        for c in (COL_VER, COL_USAR, COL_PTS, COL_ARCHIVO):
            cab.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        cab.setSectionResizeMode(COL_NOMBRE, QHeaderView.ResizeMode.Stretch)
        self._arbol.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        # Con 35 capas, colapsar a una o dos filas vuelve la lista inútil.
        self._arbol.setMinimumHeight(180)
        self._arbol.setItemDelegate(_DelegadoNombreCapa(self._arbol))
        self._arbol.itemChanged.connect(self._on_item_changed)
        self._arbol.itemSelectionChanged.connect(self._on_selection_changed)
        # La tabla se lleva todo el espacio sobrante: es el contenido principal
        # del dock y con muchas capas necesita crecer.
        layout.addWidget(self._arbol, stretch=1)

        fila = QWidget()
        fc = QHBoxLayout(fila)
        fc.setContentsMargins(0, 0, 0, 0)
        self._btn_eliminar = QPushButton("Eliminar")
        self._btn_eliminar.setToolTip(
            "Elimina las capas seleccionadas (pide confirmación).\n"
            "La última eliminación se puede deshacer.")
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
            "Funde en una sola capa las capas seleccionadas (Ctrl o Shift para "
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
        """Repuebla la tabla desde el LayerStack."""
        activas = set(getattr(stack, "active_indices", [stack.active_index]))
        self._updating = True
        try:
            self._arbol.clear()
            for i, capa in enumerate(stack):
                n = len(capa.pcd.points)
                descarte = getattr(capa, "descarte", False)
                n_archivo = getattr(capa, "n_pts_archivo", None)
                item = QTreeWidgetItem([
                    "", "",
                    f"{'[descarte] ' if descarte else ''}{capa.name}",
                    f"{n:,}", f"{n_archivo:,}" if n_archivo else "-"])
                for c in (COL_PTS, COL_ARCHIVO):
                    item.setTextAlignment(
                        c, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable
                              | Qt.ItemFlag.ItemIsEditable)
                item.setCheckState(COL_VER, Qt.CheckState.Checked if capa.visible
                                   else Qt.CheckState.Unchecked)
                item.setCheckState(COL_USAR, Qt.CheckState.Checked if i in activas
                                   else Qt.CheckState.Unchecked)
                if descarte:
                    item.setForeground(COL_NOMBRE, QColor("#8a6060"))
                item.setData(COL_NOMBRE, Qt.ItemDataRole.UserRole, capa.name)
                self._arbol.addTopLevelItem(item)
            self._nombres = [c.name for c in stack]
            self._visibles = [bool(c.visible) for c in stack]
            self._activas = [i in activas for i in range(len(self._nombres))]
            if stack.active_index >= 0:
                self._arbol.setCurrentItem(self._item(stack.active_index))
            self._actualizar_boton_eliminar()

            n_desc = stack.contar_descartes()
            self._btn_descartes.setEnabled(0 < n_desc < len(stack))
            self._btn_descartes.setText(
                f"Eliminar descartes ({n_desc})" if n_desc else "Eliminar descartes")
        finally:
            self._updating = False

    def marcar_activa(self, indice: int) -> None:
        """Compatibilidad: una sola capa activa."""
        self.marcar_activas([indice])

    def marcar_activas(self, indices) -> None:
        """Pone las casillas 'Usar' sin repoblar la tabla.

        Repoblarla borraría la selección, que es independiente y puede ser lo que
        el usuario esté construyendo para una acción del panel.
        """
        activas = set(indices)
        self._updating = True
        try:
            for i in range(self._n_filas()):
                self._item(i).setCheckState(
                    COL_USAR, Qt.CheckState.Checked if i in activas
                    else Qt.CheckState.Unchecked)
                if i < len(self._activas):
                    self._activas[i] = i in activas
        finally:
            self._updating = False

    def marcar_visibles(self, indices) -> None:
        """Pone las casillas 'Ver' sin repoblar la tabla."""
        visibles = set(indices)
        self._updating = True
        try:
            for i in range(self._n_filas()):
                self._item(i).setCheckState(
                    COL_VER, Qt.CheckState.Checked if i in visibles
                    else Qt.CheckState.Unchecked)
                if i < len(self._visibles):
                    self._visibles[i] = i in visibles
        finally:
            self._updating = False

    def set_restore_enabled(self, activo: bool) -> None:
        self._btn_restaurar.setEnabled(bool(activo))

    # ------------------------------------------------------------------ #
    # Internos                                                            #
    # ------------------------------------------------------------------ #

    def _fila_atajos(self, titulo: str, botones) -> QWidget:
        """Una fila de botones de atajo, con rótulo opcional."""
        fila = QWidget()
        caja = QHBoxLayout(fila)
        caja.setContentsMargins(0, 0, 0, 0)
        caja.setSpacing(4)
        if titulo:
            rotulo = QLabel(f"{titulo}:")
            rotulo.setStyleSheet("color: #999999;")
            rotulo.setMinimumWidth(34)
            caja.addWidget(rotulo)
        for etiqueta, ayuda, ranura in botones:
            b = QPushButton(etiqueta)
            b.setToolTip(ayuda)
            b.clicked.connect(ranura)
            caja.addWidget(b)
            self._botones[f"{titulo}/{etiqueta}"] = b
        return fila

    # ------------------------------------------------- atajos en bloque

    def _destino(self, clave: str) -> bool:
        return self._destinos[clave].isChecked()

    def _on_destino_cambiado(self) -> None:
        """Al menos un destino debe quedar activo, o los atajos no harían nada."""
        if not (self._destino("Ver") or self._destino("Usar")):
            emisor = self.sender()
            otro = "Usar" if emisor is self._destinos["Ver"] else "Ver"
            self._destinos[otro].setChecked(True)

    def _atajo_solo_esta(self) -> None:
        if not (filas := self._filas_seleccionadas()):
            return
        if self._destino("Ver"):
            self.layers_isolate_requested.emit(filas)
        if self._destino("Usar"):
            self._emit_activas(filas)

    def _atajo_todas(self) -> None:
        todas = list(range(self._n_filas()))
        if self._destino("Ver"):
            self._emit_visibilidad_todas(True)
        if self._destino("Usar"):
            self._emit_activas(todas)

    def _atajo_ninguna(self) -> None:
        """Vaciar el conjunto en uso es legítimo: se limpia para rearmar.

        Es lo que hace falta para pasar de doce capas en uso a otras tres sin
        desmarcar doce a mano. Quien recibe el conjunto vacío se encarga de
        cerrar la herramienta abierta; aquí no se decide eso.
        """
        if self._destino("Ver"):
            self._emit_visibilidad_todas(False)
        if self._destino("Usar"):
            self.marcar_activas([])
            self.layers_activated.emit([])

    def _atajo_invertir(self) -> None:
        if self._destino("Ver"):
            self.layers_invert_requested.emit()
        if self._destino("Usar"):
            self._invertir_activas()

    def _emit_activas(self, filas) -> None:
        """Sin capas activas no hay herramienta posible: no se emite vacío."""
        filas = sorted(set(filas))
        if filas:
            self.marcar_activas(filas)
            self.layers_activated.emit(filas)

    def _invertir_activas(self) -> None:
        en_uso = set(self._filas_marcadas(COL_USAR))
        self._emit_activas([i for i in range(self._n_filas()) if i not in en_uso])

    def _item(self, i: int) -> QTreeWidgetItem:
        return self._arbol.topLevelItem(i)

    def _n_filas(self) -> int:
        return self._arbol.topLevelItemCount()

    def _fila_de(self, item) -> int:
        return self._arbol.indexOfTopLevelItem(item)

    def _filas_marcadas(self, columna: int) -> list[int]:
        return [i for i in range(self._n_filas())
                if self._item(i).checkState(columna) == Qt.CheckState.Checked]

    def _on_item_changed(self, item, columna: int) -> None:
        """La columna dice qué cambió: casilla de ver, de usar, o el nombre."""
        if self._updating:
            return
        i = self._fila_de(item)
        if not 0 <= i < len(self._nombres):
            return

        if columna == COL_NOMBRE:
            nombre = item.data(COL_NOMBRE, Qt.ItemDataRole.UserRole)
            if nombre and nombre != self._nombres[i]:
                self._nombres[i] = nombre
                self.layer_renamed.emit(i, nombre)
            return

        marcada = item.checkState(columna) == Qt.CheckState.Checked
        # Si la capa tocada forma parte de una selección múltiple, el cambio se
        # aplica a todas: es lo que hace cualquier panel de capas.
        filas = self._filas_seleccionadas()
        objetivo = filas if (len(filas) > 1 and i in filas) else [i]

        if columna == COL_USAR:
            self._propagar(COL_USAR, objetivo, marcada, self._activas)
            self.layers_activated.emit(self._filas_marcadas(COL_USAR))
            return

        self._propagar(COL_VER, objetivo, marcada, self._visibles)
        if len(objetivo) > 1:
            self.layers_visibility_changed.emit(sorted(objetivo), marcada)
        else:
            self.layer_visibility_changed.emit(i, marcada)

    def _propagar(self, columna, filas, marcada, cache) -> None:
        """Lleva el estado de una casilla al resto de la selección."""
        estado = Qt.CheckState.Checked if marcada else Qt.CheckState.Unchecked
        self._updating = True
        try:
            for f in filas:
                self._item(f).setCheckState(columna, estado)
                if f < len(cache):
                    cache[f] = marcada
        finally:
            self._updating = False

    def _on_selection_changed(self) -> None:
        # La selección ya NO decide qué capas están activas: eso lo dice la
        # columna 'Usar'. Aquí solo habilita los botones que actúan sobre ella.
        self._btn_unir.setEnabled(len(self._arbol.selectedItems()) >= 2)
        self._actualizar_boton_eliminar()

    def _filas_seleccionadas(self) -> list[int]:
        return sorted(self._fila_de(it) for it in self._arbol.selectedItems())

    def _emit_isolate(self) -> None:
        if filas := self._filas_seleccionadas():
            self.layers_isolate_requested.emit(filas)

    def _emit_visibilidad_todas(self, visible: bool) -> None:
        self.layers_visibility_changed.emit(list(range(self._n_filas())), visible)

    def _emit_merge(self) -> None:
        filas = self._filas_seleccionadas()
        if len(filas) >= 2:
            self.layers_merge_requested.emit(filas)

    def _emit_removed(self) -> None:
        # La selección, no la fila actual: si el usuario marcó doce capas y el
        # botón borrara solo una, la acción no correspondería a lo que ve.
        item = self._arbol.currentItem()
        filas = self._filas_seleccionadas() or (
            [self._fila_de(item)] if item is not None else [])
        if filas:
            self.layers_removed.emit(filas)

    def _actualizar_boton_eliminar(self) -> None:
        """El botón dice cuántas capas se llevará por delante."""
        n = len(self._filas_seleccionadas())
        total = self._n_filas()
        self._btn_eliminar.setText("Eliminar" if n <= 1 else f"Eliminar {n} capas")
        # No se pueden borrar todas: siempre debe quedar una.
        self._btn_eliminar.setEnabled(total > 1 and 0 < n < total)
