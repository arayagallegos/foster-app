"""
crop_dock.py — Dock lateral con los controles de la herramienta activa.

No accede a Project ni Viewer: emite señales que MainWindow conecta.
Colores de previsualización en toda la app: VERDE = se conserva, ROJO = se elimina.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup, QDockWidget, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QMessageBox, QPushButton, QRadioButton, QSlider,
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
    ".ply — ojo: si dejas visibles la fuente Y su recorte, exportas puntos "
    "duplicados.<br><br>"
    "<b>Camara</b><br>Shift+arrastrar desplaza el encuadre (pan)."
)


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

        self._lbl_co_info = QLabel("—")
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
        self._lbl_si_conc = QLabel("Concordancia: —")
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
            self._lbl_si_conc.setText("Concordancia: —")
        else:
            self._lbl_si_conc.setText(
                f"Concordancia: {100 * conc:.1f} %  ·  relleno {n_relleno:,} pts")
        self._btn_si_aplicar.setEnabled(n_relleno > 0)

    def simetria_acota_zona(self) -> bool:
        return self._chk_si_zona.isChecked()

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
               "cono": 5, "dbscan": 6, "simetria": 7}

    def set_tool(self, tool: str) -> None:
        """Cambia la página de herramienta."""
        if tool not in self.PAGINAS:
            raise ValueError(f"Herramienta desconocida: {tool}")
        self._stack.setCurrentIndex(self.PAGINAS[tool])
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
        QMessageBox.information(self, "Ayuda — Recorte", _AYUDA)

    def _emit_lasso_started(self) -> None:
        checked = self._lasso_op_group.checkedButton()
        op = checked.property("lasso_op") if checked else "union"
        self.lasso_started.emit(op)
