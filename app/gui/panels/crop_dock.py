"""
crop_dock.py — Dock lateral con los controles de la herramienta activa.

No accede a Project ni Viewer: emite señales que MainWindow conecta.
Colores de previsualización en toda la app: VERDE = se conserva, ROJO = se elimina.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup, QDockWidget, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QRadioButton, QSlider,
    QCheckBox, QStackedWidget, QVBoxLayout, QWidget,
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
    ".ply. Ojo: si dejas visibles la fuente Y su recorte, exportas puntos "
    "duplicados.<br><br>"
    "<b>Camara</b><br>Shift+arrastrar desplaza el encuadre (pan)."
)


# Ayuda por herramienta. Un texto único obligaba a leer sobre el lazo estando en
# la esfera, que es justo cuando el usuario no necesita saber del lazo.
_COMUN = (
    "<hr><b>En todas las herramientas</b><br>"
    "Colores de previsualización: VERDE = se conserva, ROJO = se elimina.<br>"
    "Las herramientas operan sobre las capas marcadas en <b>Usar</b>, que "
    "pueden ser varias.<br>"
    "Si los manipuladores quedan detrás de la nube, baja la opacidad o el "
    "tamaño de punto desde el menú Visibilidad."
)

_AYUDA_TOOL = {
    "caja": (
        "<b>Recorte por caja</b><br>"
        "Define un prisma arrastrando las esferas de sus caras. Lo de dentro se "
        "conserva y lo de fuera va a una capa de descarte.<br><br>"
        "Se aplica a TODAS las capas visibles a la vez: con 35 escaneos, "
        "recortar de uno en uno es impracticable. Las capas que quedan enteras "
        "dentro no se dividen, y las que quedan enteras fuera solo se ocultan."
    ),
    "lazo": (
        "<b>Recorte por lazo</b><br>"
        "1. Elige la operación.<br>"
        "2. 'Iniciar lazo' y clic sobre la vista para marcar vértices.<br>"
        "3. Cierra con clic derecho, doble clic o clic sobre el primer "
        "vértice.<br>"
        "4. Encadena los lazos que necesites y pulsa 'Aplicar recorte'.<br><br>"
        + "<br>".join(f"&middot; <b>{lab}</b>: {ayuda}"
                      for lab, _v, ayuda in LASSO_OPS)
        + "<br><br><b>Cuidado:</b> el lazo selecciona por proyección en "
        "pantalla, así que captura a cualquier profundidad. Sobre una superficie "
        "curva perfora también la cara opuesta: para eso usa la agrupación, que "
        "opera en tres dimensiones."
    ),
    "esfera": (
        "<b>Primitiva esfera</b><br>"
        "Arrastra la esfera sobre la nube; al soltar se ajusta a los puntos "
        "cercanos y se previsualiza lo capturado.<br><br>"
        "<b>Tolerancia</b>: cuán lejos de la cáscara puede estar un punto "
        "capturado.<br>"
        "<b>Recorte angular</b>: en qué franja de la esfera buscar.<br><br>"
        "Los dos hacen falta y no se sustituyen. En la cúpula del Observatorio "
        "el faldón se desvía hasta 36 cm de la esfera y las compuertas están a "
        "43 cm: no existe una tolerancia que capture uno sin absorber las otras. "
        "Lo que sí los separa es el ángulo, porque no se solapan."
    ),
    "plano": (
        "<b>Primitiva plano</b><br>"
        "Coloca el plano y se ajusta a los puntos cercanos al soltarlo.<br><br>"
        "<b>Tolerancia</b>: distancia máxima al plano.<br>"
        "<b>Radio</b>: limita el disco, para no capturar material lejano que "
        "casualmente cae en el mismo plano.<br>"
        "<b>Desplazamiento</b>: mueve el plano por su normal sin reajustarlo. "
        "'Volver al ajuste' deshace el desplazamiento."
    ),
    "cilindro": (
        "<b>Primitiva cilindro</b><br>"
        "No hay manipulador de cilindro: se coloca el EJE arrastrando los dos "
        "extremos del segmento, y el radio lo recupera el ajuste.<br><br>"
        "<b>Tramo</b>: limita el trozo del eje que se usa, para capturar solo "
        "una franja de altura.<br><br>"
        "Basta una aproximación gruesa: partiendo de un eje puesto a propósito "
        "torcido, el ajuste recuperó un radio de 4,395 m frente a 4,41 m reales."
    ),
    "cono": (
        "<b>Primitiva cono</b><br>"
        "Igual que el cilindro: se coloca el eje y el ajuste recupera la "
        "apertura. Con el eje fijo, el radio de un cono crece linealmente con la "
        "altura, así que ajustar el cono se reduce a ajustar una recta.<br><br>"
        "<b>Apertura</b>: cambia el semiángulo a mano. El cono gira en torno a "
        "la mitad del tramo para no despegarse de los datos por un extremo. "
        "'Volver al ajuste' recupera el valor calculado.<br><br>"
        "El cilindro es el caso de apertura cero."
    ),
    "dbscan": (
        "<b>Agrupación por densidad</b><br>"
        "Agrupa puntos por cercanía mutua, sin suponer ninguna forma. Es para lo "
        "que las primitivas no describen: mobiliario, vegetación, "
        "contrafuertes.<br><br>"
        "<b>Va DESPUÉS de las primitivas</b>, no en su lugar. Separa lo que está "
        "físicamente desconectado, y una estructura completa es una sola "
        "superficie continua: aplicada a la nube entera devuelve un único grupo "
        "con el 99,9 % de los puntos. Retira antes las superficies grandes.<br><br>"
        "<b>Radio</b> y <b>vecinos mínimos</b> hablan de la misma densidad. "
        "Parte siempre del valor sugerido: está calculado a partir del espaciado "
        "de esta nube, y un valor traído de otra nube no se traslada.<br><br>"
        "<b>Tamaño mínimo</b> oculta los grupos diminutos, que suelen ser "
        "centenares. Se aplica después de agrupar, así que moverlo es "
        "instantáneo."
    ),
    "simetria": (
        "<b>Reparación por simetría</b><br>"
        "Completa zonas sin cobertura reflejando material realmente medido del "
        "lado opuesto. No inventa superficie a partir de un modelo: traslada "
        "puntos que existen.<br><br>"
        "1. Coloca el plano de simetría a ojo.<br>"
        "2. 'Refinar plano': el ajuste lo afina sobre los datos.<br>"
        "3. Mira la <b>concordancia</b> antes de aceptar.<br><br>"
        "<b>Concordancia</b>: fracción de puntos reflejados que caen SOBRE "
        "puntos reales. Comprueba que la simetría existe, usando la zona donde "
        "hay material a ambos lados. Alta significa que el plano está bien "
        "puesto; baja, que el relleno sería material inventado sin fundamento. "
        "En el Observatorio: 91-95 % en las compuertas, 88 % en la cúpula.<br><br>"
        "El relleno va a una CAPA APARTE y se dibuja en azul, no en verde: es "
        "material generado y debe poder distinguirse del medido. Únelo a la "
        "entidad solo cuando estés conforme."
    ),
    "malla": (
        "<b>Reconstrucción de malla</b><br>"
        "Convierte la entidad en superficie triangulada exportable. Es el último "
        "paso, sobre una entidad ya segmentada y reparada.<br><br>"
        "<b>Perímetro máximo de triángulo</b>: impide que la triangulación "
        "puentee los vacíos con triángulos estirados. Es el parámetro que decide "
        "el resultado; parte del valor sugerido, que son 14 veces el espaciado "
        "de esta nube.<br><br>"
        "<b>Cerrar agujeros</b>: los pequeños son oclusiones y conviene "
        "cerrarlos; los grandes son zonas sin cobertura, y cerrarlos sería "
        "fabricar superficie.<br>"
        "<hr>"
        "<b>Cómo leer las cifras</b><br>"
        "La malla se compara con los puntos en dos direcciones opuestas, y hay "
        "que mirar las dos porque responden preguntas distintas.<br><br>"

        "<b>Cobertura.</b> Dice si <b>falta</b> superficie. Para cada punto "
        "medido, mira a qué distancia le queda la superficie: si la malla dejó "
        "fuera una zona donde sí había datos, esos puntos quedan lejos y la "
        "cifra sube.<br>"
        "&nbsp;&nbsp;&middot; <b>Resolución.</b> Es el mínimo que la cobertura "
        "puede dar. La distancia se mide contra una versión muestreada de la "
        "superficie, así que nunca sale cero exacto. <i>Ejemplo:</i> cobertura "
        "0,84 cm con resolución 0,82 cm significa que la malla pasa por los "
        "puntos tan cerca como puede medirse, y no hay nada que corregir.<br>"
        "&nbsp;&nbsp;&middot; <b>Peor 5 %.</b> La media esconde los casos malos: "
        "si 100 puntos de 10.000 están a 5 cm, apenas se mueve. Esta cifra los "
        "delata.<br><br>"

        "<b>Invención.</b> Dice si <b>sobra</b> superficie: qué porcentaje de la "
        "malla está a más de 5 cm de cualquier punto medido, es decir, "
        "superficie que nadie escaneó y el algoritmo se inventó.<br>"
        "&nbsp;&nbsp;Mirar solo la cobertura engañaría: hay métodos que dan "
        "1,75 cm de cobertura (aparentemente excelente) fabricando el 31 % de la "
        "superficie.<br><br>"

        "<b>Contornos.</b> Dicen qué quedó <b>abierto</b>. Un agujero es un "
        "contorno cerrado de borde, no un hueco visual: el borde exterior de la "
        "pieza cuenta como UNO por dentado que se vea. Su perímetro mide cuánto "
        "<i>serpentea</i> ese borde, no el tamaño del hueco, y por eso salen "
        "decenas de metros en piezas pequeñas.<br>"
        "&nbsp;&nbsp;<b>El mayor</b> te dice a cuánto habría que subir el umbral "
        "para cerrarlos todos, y por tanto cuánta superficie fabricarías."
    ),
}


class CropDock(QDockWidget):
    """Dock con los controles de la herramienta activa."""

    # Herramienta caja
    tool_apply = pyqtSignal()
    tool_cancel = pyqtSignal()
    # Herramienta lazo
    # (tolerancia, phi_min, phi_max)
    esfera_params_changed = pyqtSignal(float, float, float)
    esfera_capturar = pyqtSignal()

    plano_params_changed = pyqtSignal(float, float, float)  # tol, radio, offset
    plano_capturar = pyqtSignal()
    plano_reajustar = pyqtSignal()    # el radio cambió: hay que re-ajustar

    cilindro_params_changed = pyqtSignal(float, float, float)  # tol, th_min, th_max
    cilindro_capturar = pyqtSignal()

    # tol, th_min, th_max, apertura (semiángulo en grados, con signo)
    cono_params_changed = pyqtSignal(float, float, float, float)
    cono_capturar = pyqtSignal()
    cono_reajustar = pyqtSignal()   # volver a la apertura ajustada

    dbscan_agrupar = pyqtSignal(float, int)      # eps, min_points
    dbscan_eps_changed = pyqtSignal(float)       # para recalcular la referencia
    dbscan_umbral_changed = pyqtSignal(int)      # tamaño mínimo de cluster
    dbscan_seleccion_changed = pyqtSignal(list)  # ids de cluster marcados
    dbscan_capturar = pyqtSignal()
    dbscan_eliminar = pyqtSignal()

    malla_generar = pyqtSignal(float, bool, float)   # perimetro, rellenar, per. relleno
    malla_exportar = pyqtSignal()
    malla_vista_changed = pyqtSignal(str)   # 'malla' | 'nube' | 'ambas'

    simetria_params_changed = pyqtSignal(float, bool)  # tolerancia, acotar zona
    simetria_refinar = pyqtSignal()
    simetria_aplicar = pyqtSignal()


    lasso_started = pyqtSignal(str)   # set_op
    lasso_apply = pyqtSignal()
    lasso_cancel = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Herramienta", parent)
        self.setObjectName("tool_dock")
        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI                                                                 #
    # ------------------------------------------------------------------ #

    _ma_sugerido: float | None = None
    _tool_actual: str | None = None

    def _build_ui(self) -> None:
        contenido = QWidget()
        layout = QVBoxLayout(contenido)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Encabezado: solo el boton de ayuda (toda la explicacion vive ahi)
        fila_top = QWidget()
        ft = QHBoxLayout(fila_top)
        ft.setContentsMargins(0, 0, 0, 0)
        btn_ayuda = QPushButton("Ayuda")
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
        self._stack.addWidget(self._build_page_plano())    # index 3
        self._stack.addWidget(self._build_page_cilindro())  # index 4
        self._stack.addWidget(self._build_page_cono())      # index 5
        self._stack.addWidget(self._build_page_dbscan())    # index 6
        self._stack.addWidget(self._build_page_simetria())  # index 7
        self._stack.addWidget(self._build_page_malla())      # index 8
        tool_layout.addWidget(self._stack)
        layout.addWidget(self._tool_box)

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

    def _build_page_plano(self) -> QWidget:
        """Controles del plano: tolerancia, radio del disco y desplazamiento.

        El RADIO es indispensable: un plano es infinito, y sin acotarlo la
        captura se lleva cualquier superficie coplanar del resto de la nube.
        Además decide qué puntos entran al ajuste, no solo cuáles se capturan.
        """
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("Coloca el plano sobre la superficie.\n"
                             "Al soltar se ajusta y muestra lo capturado."))

        self._lbl_pl_tol = QLabel()
        self._sld_pl_tol = QSlider(Qt.Orientation.Horizontal)
        self._sld_pl_tol.setRange(2, 40)             # 0.02 .. 0.40 m
        self._sld_pl_tol.setValue(8)
        lay.addWidget(self._lbl_pl_tol)
        lay.addWidget(self._sld_pl_tol)

        self._lbl_pl_radio = QLabel()
        self._sld_pl_radio = QSlider(Qt.Orientation.Horizontal)
        self._sld_pl_radio.setRange(1, 100)          # 0.5 .. 50 m
        self._sld_pl_radio.setValue(10)
        lay.addWidget(self._lbl_pl_radio)
        lay.addWidget(self._sld_pl_radio)

        # Arrastrar el widget de VTK sirve para orientar, no para desplazar con
        # precisión. Este slider mueve el plano a lo largo de su normal.
        self._lbl_pl_off = QLabel()
        self._sld_pl_off = QSlider(Qt.Orientation.Horizontal)
        self._sld_pl_off.setRange(-500, 500)          # -5.00 .. +5.00 m
        self._sld_pl_off.setValue(0)
        lay.addWidget(self._lbl_pl_off)
        lay.addWidget(self._sld_pl_off)

        self._btn_pl_off_cero = QPushButton("Volver al ajuste")
        self._btn_pl_off_cero.setToolTip(
            "Devuelve el plano a donde lo dejó el ajuste a los datos.")
        self._btn_pl_off_cero.clicked.connect(lambda: self._sld_pl_off.setValue(0))
        lay.addWidget(self._btn_pl_off_cero)

        for s in (self._sld_pl_tol, self._sld_pl_radio, self._sld_pl_off):
            s.valueChanged.connect(self._emit_plano_params)
        # El re-ajuste cuesta ~300 ms sobre 2 M de puntos: se dispara AL SOLTAR,
        # no en cada paso del arrastre, igual que el ajuste de la esfera.
        self._sld_pl_radio.sliderReleased.connect(self.plano_reajustar)

        self._btn_capturar_plano = QPushButton("Capturar entidad")
        self._btn_capturar_plano.setEnabled(False)
        self._btn_capturar_plano.clicked.connect(self.plano_capturar)
        lay.addWidget(self._btn_capturar_plano)

        self._actualizar_labels_plano()
        return page

    def _actualizar_labels_plano(self) -> None:
        self._lbl_pl_tol.setText(f"Tolerancia: {self._sld_pl_tol.value() / 100:.2f} m")
        self._lbl_pl_radio.setText(f"Radio del disco: {self._sld_pl_radio.value() / 2:.1f} m")
        self._lbl_pl_off.setText(
            f"Desplazar por la normal: {self._sld_pl_off.value() / 100:+.2f} m")

    def _emit_plano_params(self) -> None:
        self._actualizar_labels_plano()
        self.plano_params_changed.emit(
            self._sld_pl_tol.value() / 100.0,
            self._sld_pl_radio.value() / 2.0,
            self._sld_pl_off.value() / 100.0,
        )

    def reset_plano_offset(self) -> None:
        """Tras un ajuste nuevo el desplazamiento vuelve a cero: se mide desde
        donde quedó el plano ajustado, no desde el anterior."""
        bloqueado = self._sld_pl_off.blockSignals(True)
        self._sld_pl_off.setValue(0)
        self._sld_pl_off.blockSignals(bloqueado)
        self._actualizar_labels_plano()

    def set_plano_has_selection(self, has: bool) -> None:
        self._btn_capturar_plano.setEnabled(bool(has))

    def _build_page_cilindro(self) -> QWidget:
        """Controles del cilindro: tolerancia, tramo de eje y sector angular.

        Un cilindro es infinito a lo largo de su eje: sin el TRAMO, capturar el
        tambor se lleva lo que esté alineado más arriba. El SECTOR permite
        quedarse con media cáscara, igual que el recorte angular de la esfera.
        """
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("Arrastra los extremos de la línea para dar el eje.\n"
                             "Al soltar se ajusta el radio a los datos."))


        self._lbl_ci_tol = QLabel()
        self._sld_ci_tol = QSlider(Qt.Orientation.Horizontal)
        self._sld_ci_tol.setRange(2, 40)
        self._sld_ci_tol.setValue(8)
        lay.addWidget(self._lbl_ci_tol)
        lay.addWidget(self._sld_ci_tol)

        self._lbl_ci_theta = QLabel()
        self._sld_ci_th_min = QSlider(Qt.Orientation.Horizontal)
        self._sld_ci_th_min.setRange(0, 360)
        self._sld_ci_th_min.setValue(0)
        self._sld_ci_th_max = QSlider(Qt.Orientation.Horizontal)
        self._sld_ci_th_max.setRange(0, 360)
        self._sld_ci_th_max.setValue(360)
        lay.addWidget(self._lbl_ci_theta)
        lay.addWidget(self._sld_ci_th_min)
        lay.addWidget(self._sld_ci_th_max)

        for s in (self._sld_ci_tol, self._sld_ci_th_min, self._sld_ci_th_max):
            s.valueChanged.connect(self._emit_cilindro_params)

        self._btn_capturar_cil = QPushButton("Capturar entidad")
        self._btn_capturar_cil.setEnabled(False)
        self._btn_capturar_cil.clicked.connect(self.cilindro_capturar)
        lay.addWidget(self._btn_capturar_cil)

        self._actualizar_labels_cilindro()
        return page

    def _actualizar_labels_cilindro(self) -> None:
        self._lbl_ci_tol.setText(f"Tolerancia: {self._sld_ci_tol.value() / 100:.2f} m")
        self._lbl_ci_theta.setText(
            f"Sector angular: {self._sld_ci_th_min.value()}° – "
            f"{self._sld_ci_th_max.value()}°")

    def _emit_cilindro_params(self) -> None:
        self._actualizar_labels_cilindro()
        self.cilindro_params_changed.emit(
            self._sld_ci_tol.value() / 100.0,
            float(self._sld_ci_th_min.value()),
            float(self._sld_ci_th_max.value()),
        )

    def set_cilindro_has_selection(self, has: bool) -> None:
        self._btn_capturar_cil.setEnabled(bool(has))

    def _build_page_cono(self) -> QWidget:
        """Controles del cono: tolerancia y sector angular.

        El tramo de eje sale de la línea, igual que en el cilindro. En el cono
        importa más que en aquel: fuera del tramo medido la superficie se abre o
        se cierra hacia el vértice, y capturaría cosas que no tienen relación.
        """
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("Arrastra los extremos de la línea para dar el eje.\n"
                             "Al soltar se ajusta la apertura a los datos."))

        self._lbl_co_tol = QLabel()
        self._sld_co_tol = QSlider(Qt.Orientation.Horizontal)
        self._sld_co_tol.setRange(2, 40)
        self._sld_co_tol.setValue(8)
        lay.addWidget(self._lbl_co_tol)
        lay.addWidget(self._sld_co_tol)

        self._lbl_co_theta = QLabel()
        self._sld_co_th_min = QSlider(Qt.Orientation.Horizontal)
        self._sld_co_th_min.setRange(0, 360)
        self._sld_co_th_min.setValue(0)
        self._sld_co_th_max = QSlider(Qt.Orientation.Horizontal)
        self._sld_co_th_max.setRange(0, 360)
        self._sld_co_th_max.setValue(360)
        lay.addWidget(self._lbl_co_theta)
        lay.addWidget(self._sld_co_th_min)
        lay.addWidget(self._sld_co_th_max)

        # La línea solo da el EJE; la apertura la decide el ajuste. Sin este
        # control el usuario no puede corregirlo, y cuando el ajuste sale plano
        # el cono se ve idéntico a un cilindro sin manera de arreglarlo.
        self._lbl_co_ap = QLabel()
        self._sld_co_ap = QSlider(Qt.Orientation.Horizontal)
        self._sld_co_ap.setRange(-800, 800)        # -80.0 .. +80.0 grados
        self._sld_co_ap.setValue(0)
        self._sld_co_ap.setToolTip(
            "Semiángulo de apertura. Gira la superficie en torno a la mitad del "
            "tramo, así que el cono no se despega de los datos al ajustarlo.")
        lay.addWidget(self._lbl_co_ap)
        lay.addWidget(self._sld_co_ap)

        self._btn_co_ap_reset = QPushButton("Volver al ajuste")
        self._btn_co_ap_reset.setToolTip(
            "Devuelve la apertura a la que encontró RANSAC en los datos.")
        self._btn_co_ap_reset.clicked.connect(self.cono_reajustar)
        lay.addWidget(self._btn_co_ap_reset)

        for s in (self._sld_co_tol, self._sld_co_th_min, self._sld_co_th_max,
                  self._sld_co_ap):
            s.valueChanged.connect(self._emit_cono_params)

        self._lbl_co_info = QLabel("Sin ajustar")
        self._lbl_co_info.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        lay.addWidget(self._lbl_co_info)

        self._btn_capturar_cono = QPushButton("Capturar entidad")
        self._btn_capturar_cono.setEnabled(False)
        self._btn_capturar_cono.clicked.connect(self.cono_capturar)
        lay.addWidget(self._btn_capturar_cono)

        self._actualizar_labels_cono()
        return page

    def _actualizar_labels_cono(self) -> None:
        self._lbl_co_tol.setText(f"Tolerancia: {self._sld_co_tol.value() / 100:.2f} m")
        self._lbl_co_theta.setText(
            f"Sector angular: {self._sld_co_th_min.value()}° – "
            f"{self._sld_co_th_max.value()}°")
        self._lbl_co_ap.setText(
            f"Apertura (semiángulo): {self._sld_co_ap.value() / 10:+.1f}°")

    def _emit_cono_params(self) -> None:
        self._actualizar_labels_cono()
        self.cono_params_changed.emit(
            self._sld_co_tol.value() / 100.0,
            float(self._sld_co_th_min.value()),
            float(self._sld_co_th_max.value()),
            self._sld_co_ap.value() / 10.0,
        )

    def set_cono_apertura(self, grados: float) -> None:
        """Refleja en el slider la apertura que encontró el ajuste, sin emitir:
        el valor mostrado tiene que ser el que está en uso."""
        bloqueado = self._sld_co_ap.blockSignals(True)
        self._sld_co_ap.setValue(int(round(max(-80.0, min(80.0, grados)) * 10)))
        self._sld_co_ap.blockSignals(bloqueado)
        self._actualizar_labels_cono()

    def set_cono_has_selection(self, has: bool) -> None:
        self._btn_capturar_cono.setEnabled(bool(has))

    def set_cono_info(self, texto: str) -> None:
        self._lbl_co_info.setText(texto)

    def _build_page_dbscan(self) -> QWidget:
        """Agrupación por densidad: limpiar ruido y extraer lo no primitivo.

        Una sola herramienta para las dos cosas. DBSCAN es la parte cara
        (segundos o minutos), y separarla en dos herramientas obligaría a
        correrlo y afinarlo dos veces para acabar mirando la misma lista de
        clusters. Lo que cambia entre limpiar y segmentar es solo el botón
        final: Eliminar o Capturar.
        """
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)

        self._lbl_db_ref = QLabel("Carga una nube para ver el espaciado.")
        self._lbl_db_ref.setWordWrap(True)
        self._lbl_db_ref.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        lay.addWidget(self._lbl_db_ref)

        self._lbl_db_eps = QLabel()
        self._sld_db_eps = QSlider(Qt.Orientation.Horizontal)
        self._sld_db_eps.setRange(2, 100)          # 0.02 .. 1.00 m
        self._sld_db_eps.setValue(15)
        self._sld_db_eps.setToolTip(
            "Distancia máxima para considerar dos puntos vecinos. Por debajo del "
            "espaciado de la nube todo queda como ruido; muy por encima, la "
            "estructura entera se funde en un solo grupo.")
        lay.addWidget(self._lbl_db_eps)
        lay.addWidget(self._sld_db_eps)

        self._lbl_db_min = QLabel()
        self._sld_db_min = QSlider(Qt.Orientation.Horizontal)
        self._sld_db_min.setRange(3, 100)
        self._sld_db_min.setValue(20)
        self._sld_db_min.setToolTip(
            "Vecinos que necesita un punto para considerarse parte de un núcleo "
            "denso.")
        lay.addWidget(self._lbl_db_min)
        lay.addWidget(self._sld_db_min)

        for s in (self._sld_db_eps, self._sld_db_min):
            s.valueChanged.connect(self._actualizar_labels_dbscan)
        self._sld_db_eps.valueChanged.connect(
            lambda v: self.dbscan_eps_changed.emit(v / 100.0))

        # Botón explícito: agrupar es lo caro, no puede dispararse al arrastrar
        self._btn_db_run = QPushButton("Agrupar")
        self._btn_db_run.setToolTip(
            "Ejecuta DBSCAN. Es la operación cara: se lanza a mano, no al mover "
            "los sliders.")
        self._btn_db_run.clicked.connect(
            lambda: self.dbscan_agrupar.emit(self._sld_db_eps.value() / 100.0,
                                             self._sld_db_min.value()))
        lay.addWidget(self._btn_db_run)

        self._lbl_db_umbral = QLabel()
        self._sld_db_umbral = QSlider(Qt.Orientation.Horizontal)
        self._sld_db_umbral.setRange(0, 5000)
        self._sld_db_umbral.setValue(0)
        self._sld_db_umbral.setToolTip(
            "Los clusters con menos puntos que este umbral pasan a ruido. Se "
            "aplica después de agrupar, así que moverlo es instantáneo.")
        self._sld_db_umbral.valueChanged.connect(self._emit_dbscan_umbral)
        lay.addWidget(self._lbl_db_umbral)
        lay.addWidget(self._sld_db_umbral)

        self._lista_clusters = QListWidget()
        self._lista_clusters.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection)
        self._lista_clusters.setToolTip(
            "Ctrl o Shift para marcar varios. Lo marcado se resalta en el visor.")
        self._lista_clusters.setMinimumHeight(140)
        self._lista_clusters.itemSelectionChanged.connect(
            self._emit_dbscan_seleccion)
        lay.addWidget(self._lista_clusters)

        fila = QWidget()
        fl = QHBoxLayout(fila)
        fl.setContentsMargins(0, 0, 0, 0)
        self._btn_db_capturar = QPushButton("Capturar")
        self._btn_db_capturar.setToolTip(
            "Cada cluster marcado pasa a ser su propia capa, lista para "
            "renombrar. Después puedes unir las que sean la misma entidad.")
        self._btn_db_capturar.setEnabled(False)
        self._btn_db_capturar.clicked.connect(self.dbscan_capturar)
        self._btn_db_eliminar = QPushButton("Eliminar")
        self._btn_db_eliminar.setToolTip(
            "Quita de la capa los clusters marcados. Para limpiar ruido y "
            "objetos que no son estructura.")
        self._btn_db_eliminar.setEnabled(False)
        self._btn_db_eliminar.clicked.connect(self.dbscan_eliminar)
        fl.addWidget(self._btn_db_capturar)
        fl.addWidget(self._btn_db_eliminar)
        lay.addWidget(fila)

        self._actualizar_labels_dbscan()
        return page

    def _actualizar_labels_dbscan(self) -> None:
        self._lbl_db_eps.setText(f"eps (radio de vecindad): "
                                 f"{self._sld_db_eps.value() / 100:.2f} m")
        self._lbl_db_min.setText(f"min_points: {self._sld_db_min.value()}")
        v = self._sld_db_umbral.value()
        self._lbl_db_umbral.setText(
            f"Tamaño mínimo de cluster: {v:,} pts" if v else
            "Tamaño mínimo de cluster: sin filtro")

    def _emit_dbscan_umbral(self) -> None:
        self._actualizar_labels_dbscan()
        self.dbscan_umbral_changed.emit(self._sld_db_umbral.value())

    def _emit_dbscan_seleccion(self) -> None:
        ids = self.seleccion_clusters()
        self._btn_db_capturar.setEnabled(bool(ids))
        self._btn_db_eliminar.setEnabled(bool(ids))
        self.dbscan_seleccion_changed.emit(ids)

    def seleccion_clusters(self) -> list:
        return [it.data(Qt.ItemDataRole.UserRole)
                for it in self._lista_clusters.selectedItems()]

    def set_dbscan_eps(self, metros: float) -> None:
        """Deja el slider en un eps adecuado para ESTA nube.

        El valor correcto depende de la densidad, que cambia con cada capa y con
        cada re-cacheo. Un valor por defecto fijo obliga al usuario a descubrir
        a mano que 0.15 m funde toda una nube densa en un solo cluster.
        """
        bloqueado = self._sld_db_eps.blockSignals(True)
        self._sld_db_eps.setValue(int(round(max(0.02, min(1.0, metros)) * 100)))
        self._sld_db_eps.blockSignals(bloqueado)
        self._actualizar_labels_dbscan()
        self.dbscan_eps_changed.emit(self._sld_db_eps.value() / 100.0)

    def set_dbscan_referencia(self, texto: str) -> None:
        self._lbl_db_ref.setText(texto)

    def refresh_clusters(self, infos) -> None:
        """Repuebla la lista. `infos` son InfoCluster ya ordenados por tamaño."""
        bloqueado = self._lista_clusters.blockSignals(True)
        self._lista_clusters.clear()
        for i in infos:
            e = i.extension
            item = QListWidgetItem(
                f"#{i.id}  ·  {i.n_pts:,} pts  ·  plano {i.planaridad:.2f}  ·  "
                f"{e[0]:.1f}×{e[1]:.1f}×{e[2]:.1f} m")
            item.setData(Qt.ItemDataRole.UserRole, i.id)
            self._lista_clusters.addItem(item)
        self._lista_clusters.blockSignals(bloqueado)
        self._btn_db_capturar.setEnabled(False)
        self._btn_db_eliminar.setEnabled(False)

    def _build_page_simetria(self) -> QWidget:
        """Reparación por reflexión especular.

        Completa zonas mal escaneadas reflejando material real del lado bien
        cubierto. Es REPARACIÓN: la salida sigue siendo una nube de puntos, no
        una malla.
        """
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("Coloca el plano de simetría más o menos donde va\n"
                             "y pulsa Refinar."))

        # La concordancia es el indicador de confianza: va destacada, porque es
        # lo que dice si el material generado tiene respaldo o es invención.
        self._lbl_si_conc = QLabel("Concordancia: sin calcular")
        self._lbl_si_conc.setStyleSheet("font-size: 15px; font-weight: 600;")
        lay.addWidget(self._lbl_si_conc)
        self._lbl_si_ayuda = QLabel(
            "Fracción de los reflejos que cae sobre puntos reales. Alta = la "
            "simetría existe y el plano es correcto.")
        self._lbl_si_ayuda.setWordWrap(True)
        self._lbl_si_ayuda.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        lay.addWidget(self._lbl_si_ayuda)

        self._lbl_si_tol = QLabel()
        self._sld_si_tol = QSlider(Qt.Orientation.Horizontal)
        self._sld_si_tol.setRange(1, 30)          # 0.01 .. 0.30 m
        self._sld_si_tol.setValue(5)
        self._sld_si_tol.setToolTip(
            "A qué distancia un reflejo se considera encima de un punto real.")
        lay.addWidget(self._lbl_si_tol)
        lay.addWidget(self._sld_si_tol)

        self._chk_si_zona = QCheckBox("Acotar a una caja")
        self._chk_si_zona.setToolTip(
            "Limita dónde se aplica la simetría. Sin acotar, una estructura "
            "simétrica solo en parte genera material falso en el resto.")
        lay.addWidget(self._chk_si_zona)

        self._sld_si_tol.valueChanged.connect(self._emit_simetria_params)
        self._chk_si_zona.toggled.connect(self._emit_simetria_params)

        self._btn_si_refinar = QPushButton("Refinar plano")
        self._btn_si_refinar.setToolTip(
            "Ajusta el plano maximizando la concordancia. Puede tardar.")
        self._btn_si_refinar.clicked.connect(self.simetria_refinar)
        lay.addWidget(self._btn_si_refinar)

        self._btn_si_aplicar = QPushButton("Crear capa de relleno")
        self._btn_si_aplicar.setToolTip(
            "Los puntos generados van a una capa aparte, para poder revisarlos "
            "o descartarlos. Únelos a la entidad cuando estés conforme.")
        self._btn_si_aplicar.setEnabled(False)
        self._btn_si_aplicar.clicked.connect(self.simetria_aplicar)
        lay.addWidget(self._btn_si_aplicar)

        self._actualizar_labels_simetria()
        return page

    def _actualizar_labels_simetria(self) -> None:
        self._lbl_si_tol.setText(
            f"Tolerancia: {self._sld_si_tol.value() / 100:.2f} m")

    def _emit_simetria_params(self) -> None:
        self._actualizar_labels_simetria()
        self.simetria_params_changed.emit(self._sld_si_tol.value() / 100.0,
                                          self._chk_si_zona.isChecked())

    def set_simetria_concordancia(self, conc: float | None, n_relleno: int = 0) -> None:
        if conc is None:
            self._lbl_si_conc.setText("Concordancia: sin calcular")
        else:
            self._lbl_si_conc.setText(
                f"Concordancia: {100 * conc:.1f} %  ·  relleno {n_relleno:,} pts")
        self._btn_si_aplicar.setEnabled(n_relleno > 0)

    def simetria_acota_zona(self) -> bool:
        return self._chk_si_zona.isChecked()

    # ------------------------------------------------------------------ malla

    def _build_page_malla(self) -> QWidget:
        """Reconstrucción: de la entidad segmentada a superficie triangulada.

        El único parámetro que gobierna el resultado es el límite de perímetro,
        y viene propuesto en proporción al espaciado de la nube (14 veces), que
        es la regla que salió del barrido. Se deja modificable porque una
        entidad con densidad muy irregular puede necesitar otro valor, pero el
        usuario no debería tener que calcularlo.
        """
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)

        self._lbl_ma_ref = QLabel("Marca una entidad en 'Usar' para empezar.")
        self._lbl_ma_ref.setWordWrap(True)
        self._lbl_ma_ref.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        lay.addWidget(self._lbl_ma_ref)

        self._lbl_ma_per = QLabel()
        self._sld_ma_per = QSlider(Qt.Orientation.Horizontal)
        self._sld_ma_per.setRange(1, 200)             # 0,01 .. 2,00 m
        self._sld_ma_per.setValue(30)
        self._sld_ma_per.setToolTip(
            "Perímetro máximo de un triángulo. Es lo que impide que la "
            "triangulación puentee los vacíos con triángulos estirados e invente "
            "superficie donde no hubo medición. Demasiado pequeño, la malla sale "
            "agujereada.")
        lay.addWidget(self._lbl_ma_per)
        lay.addWidget(self._sld_ma_per)

        self._btn_ma_sugerido = QPushButton("Volver al valor sugerido")
        self._btn_ma_sugerido.setToolTip(
            "Devuelve el límite a 14 veces el espaciado de la nube.")
        self._btn_ma_sugerido.clicked.connect(self._volver_a_perimetro_sugerido)
        lay.addWidget(self._btn_ma_sugerido)

        self._chk_ma_rellenar = QCheckBox("Cerrar agujeros pequeños")
        self._chk_ma_rellenar.setChecked(True)
        self._chk_ma_rellenar.setToolTip(
            "Los agujeros pequeños son oclusiones y conviene cerrarlos; los "
            "grandes son zonas sin cobertura y cerrarlos sería fabricar "
            "superficie. Solo se cierran los que quedan bajo el umbral.")
        self._chk_ma_rellenar.toggled.connect(self._actualizar_labels_malla)
        lay.addWidget(self._chk_ma_rellenar)

        self._lbl_ma_rel = QLabel()
        self._sld_ma_rel = QSlider(Qt.Orientation.Horizontal)
        self._sld_ma_rel.setRange(1, 200)             # 0,1 .. 20,0 m
        self._sld_ma_rel.setValue(50)
        self._sld_ma_rel.setToolTip(
            "Perímetro máximo de los agujeros que se cierran. Los mayores se "
            "dejan abiertos a propósito.")
        lay.addWidget(self._lbl_ma_rel)
        lay.addWidget(self._sld_ma_rel)

        for s in (self._sld_ma_per, self._sld_ma_rel):
            s.valueChanged.connect(self._actualizar_labels_malla)

        # Botón explícito, como en la agrupación: triangular es lo caro y no
        # puede dispararse al arrastrar un control.
        self._btn_ma_run = QPushButton("Generar malla")
        self._btn_ma_run.setToolTip(
            "Triangula las capas en uso y mide la fidelidad del resultado. "
            "Corre en segundo plano.")
        self._btn_ma_run.clicked.connect(
            lambda: self.malla_generar.emit(
                self._sld_ma_per.value() / 100.0,
                self._chk_ma_rellenar.isChecked(),
                self._sld_ma_rel.value() / 10.0))
        lay.addWidget(self._btn_ma_run)

        # Qué se mira tras generar. Hace falta porque la malla interpola los
        # puntos: dibujadas a la vez, la nube tapa la superficie por completo y
        # el resultado parece idéntico a no haber hecho nada.
        fila_vista = QWidget()
        fv = QHBoxLayout(fila_vista)
        fv.setContentsMargins(0, 0, 0, 0)
        fv.addWidget(QLabel("Ver:"))
        self._grp_ma_vista = QButtonGroup(self)
        for etiqueta, clave, ayuda in (
            ("Malla", "malla", "Solo la superficie generada."),
            ("Nube", "nube", "Solo los puntos de partida."),
            ("Ambas", "ambas",
             "La malla semitransparente con los puntos encima. Util para ver "
             "si algun punto queda fuera de la superficie."),
        ):
            rb = QRadioButton(etiqueta)
            rb.setToolTip(ayuda)
            rb.setProperty("clave", clave)
            rb.setChecked(clave == "malla")
            rb.toggled.connect(self._emit_malla_vista)
            self._grp_ma_vista.addButton(rb)
            fv.addWidget(rb)
        fv.addStretch(1)
        self._fila_ma_vista = fila_vista
        fila_vista.setEnabled(False)
        lay.addWidget(fila_vista)

        self._lbl_ma_res = QLabel("Sin generar")
        self._lbl_ma_res.setWordWrap(True)
        self._lbl_ma_res.setTextFormat(Qt.TextFormat.RichText)
        self._lbl_ma_res.setStyleSheet("font-size: 11px;")
        lay.addWidget(self._lbl_ma_res)

        self._btn_ma_export = QPushButton("Exportar malla (.ply)")
        self._btn_ma_export.setToolTip(
            "Guarda la malla generada. El nombre parte del de la entidad.")
        self._btn_ma_export.setEnabled(False)
        self._btn_ma_export.clicked.connect(self.malla_exportar)
        lay.addWidget(self._btn_ma_export)

        self._actualizar_labels_malla()
        return page

    def _actualizar_labels_malla(self) -> None:
        self._lbl_ma_per.setText(
            f"Perímetro máximo de triángulo: {self._sld_ma_per.value() / 100:.2f} m")
        self._lbl_ma_rel.setText(
            f"Cerrar agujeros de hasta: {self._sld_ma_rel.value() / 10:.1f} m")
        activo = self._chk_ma_rellenar.isChecked()
        self._lbl_ma_rel.setEnabled(activo)
        self._sld_ma_rel.setEnabled(activo)

    def _volver_a_perimetro_sugerido(self) -> None:
        if self._ma_sugerido is not None:
            self._sld_ma_per.setValue(
                max(1, min(200, round(self._ma_sugerido * 100))))

    def set_malla_referencia(self, espaciado: float | None,
                             n_puntos: int = 0) -> None:
        """Propone el límite de perímetro a partir del espaciado de la entidad."""
        if espaciado is None or espaciado <= 0:
            self._ma_sugerido = None
            self._lbl_ma_ref.setText("Marca una entidad en 'Usar' para empezar.")
            return
        self._ma_sugerido = 14.0 * espaciado
        self._lbl_ma_ref.setText(
            f"{n_puntos:,} puntos · espaciado {espaciado:.4f} m · "
            f"límite sugerido {self._ma_sugerido:.2f} m (14 × el espaciado)")
        self._volver_a_perimetro_sugerido()

    def _emit_malla_vista(self, marcado: bool) -> None:
        if marcado:
            self.malla_vista_changed.emit(self.sender().property("clave"))

    def malla_vista(self) -> str:
        boton = self._grp_ma_vista.checkedButton()
        return boton.property("clave") if boton else "malla"

    def set_malla_resultado(self, texto: str | None,
                            exportable: bool = False) -> None:
        """Mensaje suelto: en curso, o el motivo de un fallo."""
        self._lbl_ma_res.setText(texto or "Sin generar")
        self._btn_ma_export.setEnabled(bool(exportable))
        self._fila_ma_vista.setEnabled(bool(exportable))

    def set_malla_metricas(self, n_caras: int, n_vertices: int, n_puntos: int,
                           fid, stats: dict | None = None) -> None:
        """Las cifras del resultado, agrupadas por la pregunta que responden.

        En un párrafo corrido no se distingue qué mide cada número ni cuáles se
        comparan entre sí. Agrupadas —tamaño, fidelidad, diagnóstico— se leen de
        un vistazo, y la resolución queda junto a la cobertura porque es su
        suelo: sin verlas juntas, una cobertura pequeña parece un error residual
        en vez de "tan cerca como esto puede medirse".
        """
        def fila(etiqueta, valor, ayuda=""):
            titulo = f' title="{ayuda}"' if ayuda else ""
            return (f'<tr><td style="color:#999"{titulo}>{etiqueta}</td>'
                    f'<td align="right"><b>{valor}</b></td></tr>')

        inventados = max(0, n_vertices - n_puntos)
        html = ['<table cellspacing="0" cellpadding="2" width="100%">']
        html.append('<tr><td colspan="2" style="color:#ffffff">'
                    '<b>Malla</b></td></tr>')
        html.append(fila("Triángulos", f"{n_caras:,}"))
        html.append(fila("Vértices", f"{n_vertices:,}",
                         "Los puntos medidos, más los que se crearon al "
                         "parchear agujeros."))
        if inventados:
            html.append(fila("· creados al parchear", f"{inventados:,}"))

        html.append('<tr><td colspan="2" style="color:#ffffff;padding-top:6px">'
                    '<b>Fidelidad</b></td></tr>')
        html.append(fila("Cobertura", f"{fid.cobertura_media * 100:.2f} cm",
                         "Distancia media de cada punto medido a la superficie. "
                         "Dice si FALTA superficie donde sí hubo datos."))
        html.append(fila("· peor 5 %", f"{fid.cobertura_p95 * 100:.2f} cm",
                         "El 95 % de los puntos está más cerca que esto."))
        html.append(fila("· resolución", f"{fid.resolucion * 100:.2f} cm",
                         "Suelo de la medida: la cobertura no puede bajar de "
                         "aquí aunque la malla fuera exacta."))
        html.append(fila("Invención", f"{fid.invencion * 100:.2f} %",
                         f"Superficie a más de {fid.tolerancia * 100:.0f} cm de "
                         "todo punto medido. Dice si SOBRA superficie."))

        if stats:
            html.append('<tr><td colspan="2" style="color:#ffffff;'
                        'padding-top:6px"><b>Diagnóstico</b></td></tr>')
            ayuda_agujero = (
                "Un agujero es un CONTORNO cerrado de borde, no un hueco "
                "visual: el borde exterior de la entidad cuenta como uno solo "
                "por dentado que se vea en pantalla.")
            n_antes = stats.get("n_antes")
            n_abiertos = stats.get("n_holes")
            if n_antes is not None and n_abiertos is not None:
                html.append(fila("Contornos al triangular", f"{n_antes}",
                                 ayuda_agujero))
                html.append(fila("· cerrados", f"{max(0, n_antes - n_abiertos)}",
                                 "Los que quedaban por debajo del umbral."))
            html.append(fila("· abiertos", f"{n_abiertos if n_abiertos is not None else 'n/d'}",
                             "Superan el umbral: zonas sin cobertura que se "
                             "dejaron abiertas a propósito. " + ayuda_agujero))
            if stats.get("mayor"):
                # Las aristas van al lado del perímetro porque este mide
                # LONGITUD DE RECORRIDO, no tamaño de abertura: un contorno de
                # 110 m con 2.300 aristas serpentea alrededor de una pieza de
                # 7 m, y sin ese dato la cifra se lee como un agujero enorme.
                aristas = stats.get("mayor_aristas")
                valor = (f"{stats['mayor']:.1f} m ({aristas:,} aristas)"
                         if aristas else f"{stats['mayor']:.1f} m")
                html.append(fila("· el mayor", valor,
                                 "Perímetro del contorno abierto más grande: la "
                                 "longitud que recorre su borde, no el tamaño "
                                 "del hueco. Un contorno con muchas aristas es "
                                 "un borde dentado, no una abertura grande. "
                                 "Subir el umbral por encima de este valor los "
                                 "cerraría todos, fabricando esa superficie."))
            html.append(fila("Cerrada", "sí" if stats.get("is_closed") else "no",
                             "Si la superficie encierra un volumen."))
        html.append("</table>")

        self._lbl_ma_res.setText("".join(html))
        self._btn_ma_export.setEnabled(True)
        self._fila_ma_vista.setEnabled(True)

    def set_malla_ocupado(self, ocupado: bool) -> None:
        self._btn_ma_run.setEnabled(not ocupado)
        self._btn_ma_run.setText("Generando…" if ocupado else "Generar malla")

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

    PAGINAS = {"caja": 0, "lazo": 1, "esfera": 2, "plano": 3, "cilindro": 4,
               "cono": 5, "dbscan": 6, "simetria": 7, "malla": 8}

    def set_tool(self, tool: str) -> None:
        """Cambia la página de herramienta."""
        if tool not in self.PAGINAS:
            raise ValueError(f"Herramienta desconocida: {tool}")
        self._stack.setCurrentIndex(self.PAGINAS[tool])
        self._tool_actual = tool
        self._tool_box.setTitle(f"Herramienta: {tool.capitalize()}")

    def set_lasso_active(self, active: bool) -> None:
        self._btn_start_lasso.setEnabled(not bool(active))
        self._btn_cancel_lasso.setEnabled(bool(active))

    def set_lasso_has_selection(self, has: bool) -> None:
        self._btn_apply_lasso.setEnabled(bool(has))

    # ------------------------------------------------------------------ #
    # Internos                                                              #
    # ------------------------------------------------------------------ #

    def _show_help(self) -> None:
        """Ayuda de la herramienta activa, no un texto único para todas."""
        tool = self._tool_actual
        texto = _AYUDA_TOOL.get(tool)
        if texto is None:
            QMessageBox.information(self, "Ayuda", _AYUDA)
            return
        QMessageBox.information(self, f"Ayuda: {tool.capitalize()}",
                                texto + _COMUN)

    def _emit_lasso_started(self) -> None:
        checked = self._lasso_op_group.checkedButton()
        op = checked.property("lasso_op") if checked else "union"
        self.lasso_started.emit(op)
