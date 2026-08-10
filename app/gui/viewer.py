"""
viewer.py — Widget de visualización 3D para nubes de puntos y mallas.

Usa PyVista (wrapper de VTK) embebido en Qt via pyvistaqt.
El viewer maneja automáticamente el downsampling para visualización:
nubes grandes se reducen para que la interacción sea fluida, pero la
nube original se conserva en el Project para operaciones posteriores.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import open3d as o3d
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from contextlib import contextmanager

from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QCursor


def _base_perpendicular(eje: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dos vectores unitarios perpendiculares al eje. Misma convención que
    `Cilindro.base()` del motor, para que el ángulo 0 coincida en ambos."""
    eje = np.asarray(eje, float)
    eje = eje / np.linalg.norm(eje)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(eje @ ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = ref - (ref @ eje) * eje
    u = u / np.linalg.norm(u)
    return u, np.cross(eje, u)


# Límite de puntos para visualización interactiva fluida
DISPLAY_MAX_POINTS = 2_000_000


class Viewer3D(QWidget):
    """
    Widget 3D que embebe un QtInteractor de PyVista.

    Responsabilidades:
    - Mostrar nubes de puntos y mallas en 3D
    - Manejar downsampling automático para fluidez
    - Colorear por RGB si disponible, por altura si no
    - Reportar el número real de puntos mostrados
    """

    # Emite el conteo de puntos mostrados tras cargar una nube
    points_displayed = pyqtSignal(int, int)  # (n_original, n_displayed)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._active_actors: list[str] = []  # nombres de actors en la escena
        self._crop_widget = None
        self._sphere_widget = None
        self._plane_widget = None
        self._line_widget = None
        # Centro compartido de la escena: todos los actores (capas, previews)
        # se dibujan restando este centro para quedar alineados.
        self._scene_center: np.ndarray | None = None
        self._dimmed_actor: list[str] | None = None   # actores atenuados por el preview
        self._camara_encuadrada = False   # se encuadra una vez, al cargar
        # apagada por defecto: la órbita es lo que se espera al abrir un visor
        self.wasd_activo = False
        self._look_habilitado = False   # mirar con botón derecho
        self._mirando = False           # botón derecho apretado ahora
        self._cursor_origen = None
        self._opacidad_nube = 1.0
        self._tamano_punto = 2.0
        # Estado del lazo (VTK-based)
        self._lasso_callback = None
        self._lasso_obs_ids: list[int] = []
        self._lasso_saved_style = None
        self._lasso_actor2d = None
        self._lasso_screen_verts: list[tuple[int, int]] = []
        self._lasso_vtk_verts: list[tuple[int, int]] = []
        self._lasso_cursor_vtk: tuple[int, int] = (0, 0)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # El QtInteractor es el viewport VTK embebido en Qt
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#1e1e1e")  # fondo gris oscuro
        self.plotter.show_axes()

        layout.addWidget(self.plotter.interactor)
        self._instalar_wasd()

    # ------------------------------------------------------------------ #
    # Carga de nubes de puntos                                             #
    # ------------------------------------------------------------------ #

    def show_cloud(
        self,
        pcd: o3d.geometry.PointCloud,
        name: str = "cloud",
        point_size: float = 8.0,
        replace: bool = True,
    ) -> int:

        n_original = len(pcd.points)

        pcd_display = _maybe_downsample(
            pcd,
            DISPLAY_MAX_POINTS
        )

        n_display = len(pcd_display.points)

        print(
            f"[viewer] mostrando {n_display:,} de {n_original:,} puntos"
        )

        self._scene_center = np.asarray(pcd_display.points).mean(axis=0)
        poly = _o3d_to_pyvista(pcd_display, center=self._scene_center)

        print(
            "PolyData:",
            poly.n_points,
            "points"
        )

        if replace and name in self._active_actors:
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass
            self._active_actors.remove(name)

        if pcd_display.has_colors():

            self.plotter.add_mesh(
                poly,
                scalars="RGB",
                rgb=True,
                point_size=point_size,
                render_points_as_spheres=True,
                style="points",
                name=name,
            )

        else:

            self.plotter.add_mesh(
                poly,
                scalars="z",
                cmap="viridis",
                point_size=point_size,
                render_points_as_spheres=True,
                style="points",
                name=name,
                show_scalar_bar=False,
            )

        self._active_actors.append(name)

        self.plotter.reset_camera()

        self.points_displayed.emit(
            n_original,
            n_display
        )

        return n_display

    def show_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        name: str = "mesh",
        color: str = "#c8a882",  # color piedra
        opacity: float = 1.0,
    ):
        """Muestra una malla 3D en el viewer."""
        poly = _o3d_mesh_to_pyvista(mesh)

        if name in self._active_actors:
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass
            self._active_actors.remove(name)

        self.plotter.add_mesh(
            poly,
            color=color,
            opacity=opacity,
            name=name,
            show_edges=False,
        )
        self._active_actors.append(name)
        self.plotter.reset_camera()

    # ------------------------------------------------------------------ #
    # Control de la escena                                                 #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Navegación WASD                                                     #
    # ------------------------------------------------------------------ #

    # (tecla, adelante, lateral, vertical). Mayúscula = paso grande, porque VTK
    # entrega la tecla ya en mayúscula cuando se pulsa con Shift.
    _TECLAS_WASD = (
        ("w", 1.0, 0.0, 0.0), ("s", -1.0, 0.0, 0.0),
        ("d", 0.0, 1.0, 0.0), ("a", 0.0, -1.0, 0.0),
        ("e", 0.0, 0.0, 1.0), ("q", 0.0, 0.0, -1.0),
    )
    PASO_WASD = 0.08          # fracción de la distancia al foco, por pulsación

    def _instalar_wasd(self) -> None:
        """Registra las teclas de navegación.

        Hay que registrarlas sí o sí: VTK ya usa w/s para wireframe/superficie y
        q/e para CERRAR la ventana. Sin sobrescribirlas, pulsar 'q' mataría el
        visor. El interruptor `wasd_activo` decide si además mueven la cámara.
        """
        for tecla, ade, lat, ver in self._TECLAS_WASD:
            for k, factor in ((tecla, 1.0), (tecla.upper(), 4.0)):
                self.plotter.add_key_event(
                    k, lambda a=ade, l=lat, v=ver, f=factor:
                        self._mover_camara(a * f, l * f, v * f)
                )

    ACTORES_PREVIEW = ("_preview_keep", "_preview_discard",
                       "_lasso_selection", "_clusters")

    def set_opacidad_nube(self, opacidad: float) -> None:
        """Transparencia de la nube: permite ver los manipuladores de una
        primitiva cuando quedan DENTRO de la estructura."""
        self._opacidad_nube = float(np.clip(opacidad, 0.05, 1.0))
        self._aplicar_estilo_nube()

    def set_tamano_punto(self, tam: float) -> None:
        """Tamaño del punto en píxeles. Puntos chicos dejan huecos entre sí y
        hacen visible la geometría de detrás."""
        self._tamano_punto = float(np.clip(tam, 1.0, 10.0))
        self._aplicar_estilo_nube()

    def _aplicar_estilo_nube(self, render: bool = True) -> None:
        """Reaplica opacidad y tamaño de punto a TODOS los actores de nube.

        Incluye los del preview (verde/rojo). Esto importa justo cuando más:
        durante un recorte es cuando hace falta ver a través de la nube, y si
        el preview se dibujara con valores fijos, bajar la opacidad no serviría
        de nada. Es también la razón de tener una sola función en vez de fijar
        los valores en cada `add_mesh`: así no hay dos verdades que se pisen.

        No redibuja geometría, solo toca propiedades del actor: es instantáneo
        aunque la nube tenga millones de puntos.
        """
        atenuados = set(self._dimmed_actor or ())
        for nombre in list(self._active_actors) + list(self.ACTORES_PREVIEW):
            actor = self.plotter.actors.get(nombre)
            if actor is None:
                continue
            prop = actor.GetProperty()
            prop.SetPointSize(self._tamano_punto)
            if nombre in atenuados:
                # capa fuente bajo un preview: se atenúa MUCHO, pero relativo a
                # lo que pidió el usuario, no a un valor absoluto
                prop.SetOpacity(self._opacidad_nube * 0.15)
            else:
                prop.SetOpacity(self._opacidad_nube)
        if render:
            self.plotter.render()

    SENS_MOUSE = 0.0035        # radianes por píxel de movimiento

    def set_mouse_look(self, activo: bool) -> None:
        """Habilita mirar en primera persona MIENTRAS se mantiene el botón derecho.

        Sin modos: apretado miras y vuelas con WASD; al soltar recuperas el
        cursor para agarrar los manipuladores de la primitiva. Es lo que hacen
        Blender, el editor de Unreal y CloudCompare, y evita el vaivén de entrar
        y salir de un modo cada vez que hay que ajustar algo — que es
        precisamente lo que más se hace en esta herramienta.

        Mientras está apretado el cursor se oculta y se recentra en cada
        movimiento, para poder girar sin que el ratón tope con el borde de la
        pantalla; al soltar vuelve a donde estaba.
        """
        activo = bool(activo)
        if activo == self._look_habilitado:
            return
        self._look_habilitado = activo
        w = self.plotter.interactor
        if activo:
            w.installEventFilter(self)
        else:
            self._terminar_look()
            w.removeEventFilter(self)

    def _centro_global(self):
        w = self.plotter.interactor
        return w.mapToGlobal(w.rect().center())

    def _iniciar_look(self) -> None:
        self._mirando = True
        self._cursor_origen = QCursor.pos()   # para devolverlo al soltar
        self.plotter.interactor.setCursor(Qt.CursorShape.BlankCursor)
        QCursor.setPos(self._centro_global())

    def _terminar_look(self) -> None:
        if not self._mirando:
            return
        self._mirando = False
        self.plotter.interactor.unsetCursor()
        if self._cursor_origen is not None:
            QCursor.setPos(self._cursor_origen)
            self._cursor_origen = None

    def eventFilter(self, obj, ev):
        if not self._look_habilitado:
            return super().eventFilter(obj, ev)
        tipo = ev.type()
        if tipo == QEvent.Type.MouseButtonPress and                 ev.button() == Qt.MouseButton.RightButton:
            self._iniciar_look()
            return True          # VTK usa el botón derecho para zoom: se anula
        if tipo == QEvent.Type.MouseButtonRelease and                 ev.button() == Qt.MouseButton.RightButton:
            self._terminar_look()
            return True
        if tipo == QEvent.Type.MouseMove and self._mirando:
            centro = self._centro_global()
            p = ev.globalPosition().toPoint()
            dx, dy = p.x() - centro.x(), p.y() - centro.y()
            if dx or dy:
                self._girar_camara(dx, dy)
                QCursor.setPos(centro)   # el recentrado da delta cero: no gira
            return True
        return super().eventFilter(obj, ev)

    def _girar_camara(self, dx: int, dy: int) -> None:
        """Gira la mirada sobre la posición de la cámara (yaw + pitch).

        A diferencia de la órbita, la cámara NO se mueve: se queda donde está y
        gira la cabeza. El giro horizontal es siempre alrededor del eje Z del
        mundo, para que la vertical del observatorio siga siendo la vertical de
        la pantalla; y el vertical se limita antes del cenit, porque justo ahí
        la dirección y el "arriba" se vuelven paralelos y la imagen daría un
        tirón.
        """
        cam = self.plotter.camera
        pos = np.array(cam.position, float)
        d = np.array(cam.focal_point, float) - pos
        dist = float(np.linalg.norm(d))
        if dist < 1e-9:
            return
        v = d / dist

        yaw = -dx * self.SENS_MOUSE
        cy, sy = np.cos(yaw), np.sin(yaw)
        v = np.array([v[0] * cy - v[1] * sy, v[0] * sy + v[1] * cy, v[2]])

        derecha = np.cross(v, (0.0, 0.0, 1.0))
        n = float(np.linalg.norm(derecha))
        if n > 1e-9:
            derecha /= n
            pitch = -dy * self.SENS_MOUSE
            cp, sp = np.cos(pitch), np.sin(pitch)
            # rotación de Rodrigues de v alrededor de `derecha`
            v = v * cp + np.cross(derecha, v) * sp + derecha * (derecha @ v) * (1 - cp)
            v /= np.linalg.norm(v)
            if abs(v[2]) > 0.995:      # casi vertical: se frena antes del tirón
                return

        cam.focal_point = pos + v * dist
        cam.up = (0.0, 0.0, 1.0)
        self.plotter.render()

    def _mover_camara(self, adelante: float, lateral: float, vertical: float) -> None:
        """Desplaza cámara y foco juntos, en el sistema de la propia cámara.

        El paso es proporcional a la distancia al foco: lejos avanza a zancadas
        y cerca de la superficie se mueve fino, que es justo cuando hace falta
        precisión para colocar una primitiva.
        """
        if not self.wasd_activo:
            return
        cam = self.plotter.camera
        pos, foco = np.array(cam.position, float), np.array(cam.focal_point, float)
        arriba = np.array(cam.up, float)
        vista = foco - pos
        dist = float(np.linalg.norm(vista))
        if dist < 1e-9:
            return
        vista /= dist
        derecha = np.cross(vista, arriba)
        n = np.linalg.norm(derecha)
        if n < 1e-9:
            return
        derecha /= n
        arriba = np.cross(derecha, vista)      # re-ortogonaliza

        paso = dist * self.PASO_WASD
        d = (vista * adelante + derecha * lateral + arriba * vertical) * paso
        cam.position = pos + d
        cam.focal_point = foco + d
        self.plotter.render()

    @contextmanager
    def camara_fija(self):
        """
        Congela el punto de vista mientras se repinta la escena.

        Los repintados que ocurren durante una interacción (mover la caja de
        recorte, arrastrar la esfera) agregan y quitan actores, y VTK puede
        reencuadrar por su cuenta. Guardar y restaurar la cámara es
        independiente de cuál sea la llamada que reencuadra, y no cuesta nada
        si no reencuadra ninguna.
        """
        try:
            pos = self.plotter.camera_position
        except Exception:
            pos = None
        try:
            yield
        finally:
            if pos is not None:
                self.plotter.camera_position = pos

    def clear(self):
        """Elimina todos los objetos de la escena."""
        self.plotter.clear()
        self._active_actors.clear()
        self._camara_encuadrada = False   # la próxima nube vuelve a encuadrarse

    def remove_actor(self, name: str):
        """Elimina un actor por nombre."""
        try:
            self.plotter.remove_actor(name)
            if name in self._active_actors:
                self._active_actors.remove(name)
        except Exception:
            pass

    def reset_camera(self):
        self.plotter.reset_camera()

    def set_view_top(self):
        self.plotter.view_xy()

    def set_view_front(self):
        self.plotter.view_xz()

    def set_view_side(self):
        self.plotter.view_yz()

    def set_view_isometric(self):
        self.plotter.view_isometric()

    def toggle_axes(self, visible: bool):
        if visible:
            self.plotter.show_axes()
        else:
            self.plotter.hide_axes()

    def start_crop_widget(
        self,
        pcd: o3d.geometry.PointCloud,
        callback,
    ) -> None:
        """
        Activa un box widget interactivo inicializado al bounding box de pcd.

        El callback se llama cada vez que el usuario mueve o redimensiona la caja.
        Los bounds en el callback estan en coordenadas mundo (mismo sistema que pcd.points).

        Args:
            pcd:      Nube sobre la que se hace el recorte.
            callback: Funcion (min_bound: np.ndarray, max_bound: np.ndarray) -> None
        """
        if self._crop_widget is not None:
            self.stop_crop_widget()

        pts = np.asarray(pcd.points)
        center = self._scene_center if self._scene_center is not None else pts.mean(axis=0)
        pts_c = pts - center
        full_min = pts_c.min(axis=0)
        full_max = pts_c.max(axis=0)
        # Caja inicial al 25% de las dimensiones totales, centrada en el origen
        half = (full_max - full_min) * 0.125  # 0.125 = 25% / 2
        mins = -half
        maxs = half
        # PyVista espera [xmin, xmax, ymin, ymax, zmin, zmax]
        bounds = [mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]]

        def _pv_callback(box):
            b = box.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
            with self.camara_fija():
                self._dibujar_caja(b, center, callback)

        self._crop_widget = self.plotter.add_box_widget(
            callback=_pv_callback,
            bounds=bounds,
        )

    def _dibujar_caja(self, b, center, callback) -> None:
        """Redibuja la caja semitransparente y avisa los bounds en coords mundo."""
        self.plotter.add_mesh(
            pv.Box(bounds=b),
            color="#4fc3f7",
            opacity=0.12,
            style="surface",
            name="_crop_box_fill",
            show_edges=True,
            edge_color="#4fc3f7",
            line_width=1,
            reset_camera=False,
        )
        min_b = np.array([b[0], b[2], b[4]]) + center
        max_b = np.array([b[1], b[3], b[5]]) + center
        callback(min_b, max_b)

    def stop_crop_widget(self) -> None:
        """Elimina el box widget y la caja de previsualización del viewer."""
        if self._crop_widget is not None:
            try:
                self._crop_widget.Off()
            except Exception:
                pass
            self._crop_widget = None
        actor = self.plotter.actors.get("_crop_box_fill")
        if actor is not None:
            self.plotter.remove_actor("_crop_box_fill")
        self.plotter.render()

    # ------------------------------------------------------------------ #
    # Herramienta de primitivas                                           #
    # ------------------------------------------------------------------ #

    def start_sphere_widget(self, pcd: o3d.geometry.PointCloud, callback) -> None:
        """
        Activa una esfera interactiva para segmentar por primitiva.

        El callback se llama al SOLTAR (no en vivo): con nubes grandes el ajuste
        continuo hace que arrastrar se sienta pesado.

        Args:
            pcd:      Nube sobre la que se segmenta.
            callback: Funcion (centro: np.ndarray, radio: float) -> None,
                      con el centro en coordenadas mundo.
        """
        if self._sphere_widget is not None:
            self.stop_sphere_widget()

        pts = np.asarray(pcd.points)
        center = self._scene_center if self._scene_center is not None else pts.mean(axis=0)
        pts_c = pts - center
        # Esfera inicial: centrada en la nube, con radio ~40% de su extension,
        # para que arranque cerca de una superficie util en vez de en un punto.
        radio0 = float(np.ptp(pts_c, axis=0).max()) * 0.40

        def _pv_callback(centro_widget, widget):
            # PyVista solo pasa el CENTRO al callback; el radio hay que sacarlo
            # del widget (por eso pass_widget=True).
            radio = float(widget.GetRadius())
            callback(np.asarray(centro_widget) + center, radio)

        self._sphere_widget = self.plotter.add_sphere_widget(
            _pv_callback,
            center=pts_c.mean(axis=0),
            radius=radio0,
            color="#4fc3f7",
            style="wireframe",
            pass_widget=True,
            test_callback=False,
            interaction_event="end",
        )

    def stop_sphere_widget(self) -> None:
        """Elimina la esfera interactiva del viewer."""
        if self._sphere_widget is not None:
            try:
                self._sphere_widget.Off()
            except Exception:
                pass
            self._sphere_widget = None
        self.plotter.render()

    def start_plane_widget(self, pcd: o3d.geometry.PointCloud, callback,
                           ajustar_inicial: bool = True):
        """
        Activa un plano interactivo (origen + normal arrastrables).

        Args:
            pcd:      Nube sobre la que se segmenta.
            callback: Función (punto: np.ndarray, normal: np.ndarray) -> None,
                      con el punto en coordenadas mundo.
        """
        if self._plane_widget is not None:
            self.stop_plane_widget()

        pts = np.asarray(pcd.points)
        center = self._scene_center if self._scene_center is not None else pts.mean(axis=0)
        pts_c = pts - center

        def _pv_callback(normal, origen):
            callback(np.asarray(origen) + center, np.asarray(normal))

        origen0 = pts_c.mean(axis=0)
        self._plane_widget = self.plotter.add_plane_widget(
            _pv_callback,
            origin=origen0,
            normal=(0.0, 0.0, 1.0),
            bounds=self._bounds_de(pts_c),
            color="#4fc3f7",
            outline_translation=False,   # el plano se mueve, su caja no
            implicit=False,              # plano finito: se ve dónde va a capturar
            interaction_event="end",
        )
        # El callback de VTK solo se dispara al SOLTAR. Con `ajustar_inicial` se
        # fuerza un primer ajuste para que los sliders tengan sobre qué operar;
        # se omite al reiniciar tras una captura, porque ahí repintar de
        # inmediato haría parecer que la entidad recién capturada quedó a medias.
        if ajustar_inicial:
            _pv_callback((0.0, 0.0, 1.0), origen0)
        return np.asarray(origen0) + center, np.array([0.0, 0.0, 1.0])

    def mostrar_disco_plano(self, centro, normal, radio: float) -> None:
        """Dibuja el disco que el plano REALMENTE captura.

        El cuadrado del widget de VTK no tiene relación con el radio de captura:
        sin este disco el usuario ve una superficie enorme y captura una mucho
        más chica, sin manera de saberlo.
        """
        c = self._scene_center if self._scene_center is not None else 0.0
        disco = pv.Disc(center=np.asarray(centro) - c, inner=0.0, outer=float(radio),
                        normal=np.asarray(normal), r_res=1, c_res=64)
        self.plotter.add_mesh(disco, color="#ffd54f", opacity=0.25,
                              name="_plano_disco", reset_camera=False)
        self.plotter.render()

    def ocultar_disco_plano(self) -> None:
        if "_plano_disco" in self.plotter.actors:
            self.plotter.remove_actor("_plano_disco")
            self.plotter.render()

    def stop_plane_widget(self) -> None:
        self.ocultar_disco_plano()
        if self._plane_widget is not None:
            try:
                self._plane_widget.Off()
            except Exception:
                pass
            self._plane_widget = None
        self.plotter.render()

    def start_line_widget(self, pcd: o3d.geometry.PointCloud, callback) -> None:
        """
        Activa una línea interactiva, que define el EJE del cilindro.

        PyVista no trae un widget de cilindro; la línea es la forma natural de
        dar un eje, y el radio se ajusta después a los datos con RANSAC.

        Args:
            callback: Función (p1: np.ndarray, p2: np.ndarray) -> None, en
                      coordenadas mundo.
        """
        if self._line_widget is not None:
            self.stop_line_widget()

        pts = np.asarray(pcd.points)
        center = self._scene_center if self._scene_center is not None else pts.mean(axis=0)
        pts_c = pts - center

        def _pv_callback(linea):
            p1 = np.asarray(linea.points[0]) + center
            p2 = np.asarray(linea.points[-1]) + center
            callback(p1, p2)

        self._line_widget = self.plotter.add_line_widget(
            _pv_callback,
            bounds=self._bounds_de(pts_c),
            color="#4fc3f7",
            interaction_event="end",
        )

    def mostrar_fantasma_cilindro(self, punto, eje, radio: float,
                                  h_min: float, h_max: float) -> None:
        """Dibuja el cilindro ajustado.

        El widget de VTK solo muestra una línea con dos esferas: el usuario tiene
        que imaginarse la superficie que va a capturar. Ver la figura de verdad
        es la diferencia entre colocarla a ojo y colocarla con criterio.
        """
        c = self._scene_center if self._scene_center is not None else 0.0
        punto, eje = np.asarray(punto, float), np.asarray(eje, float)
        altura = float(h_max - h_min)
        if not np.isfinite(altura) or altura <= 0:
            return
        centro = punto + eje * (h_min + altura / 2.0) - c
        cil = pv.Cylinder(center=centro, direction=eje, radius=float(radio),
                          height=altura, resolution=64, capping=False)
        self.plotter.add_mesh(cil, color="#ffd54f", opacity=0.20,
                              name="_cilindro_fantasma", reset_camera=False)
        self.plotter.render()

    def mostrar_fantasma_cono(self, punto, eje, r_min: float, r_max: float,
                              h_min: float, h_max: float, n: int = 64) -> None:
        """Dibuja el tronco de cono ajustado.

        Se construye a mano y no con `pv.Cone` porque este dibuja el cono
        completo desde el vértice, que puede quedar decenas de metros fuera de
        la nube; lo que interesa ver es el tramo que realmente captura.
        """
        c = self._scene_center if self._scene_center is not None else 0.0
        punto, eje = np.asarray(punto, float), np.asarray(eje, float)
        if not np.isfinite([r_min, r_max, h_min, h_max]).all() or h_max <= h_min:
            return
        u, v = _base_perpendicular(eje)
        th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        aro = np.cos(th)[:, None] * u + np.sin(th)[:, None] * v
        abajo = punto + eje * h_min + max(r_min, 0.0) * aro - c
        arriba = punto + eje * h_max + max(r_max, 0.0) * aro - c

        pts = np.vstack([abajo, arriba])
        caras = []
        for i in range(n):
            j = (i + 1) % n
            caras += [4, i, j, n + j, n + i]
        malla = pv.PolyData(pts, faces=np.array(caras))
        self.plotter.add_mesh(malla, color="#ffd54f", opacity=0.20,
                              name="_cono_fantasma", reset_camera=False)
        self.plotter.render()

    def ocultar_fantasma_cono(self) -> None:
        if "_cono_fantasma" in self.plotter.actors:
            self.plotter.remove_actor("_cono_fantasma")
            self.plotter.render()

    def ocultar_fantasma_cilindro(self) -> None:
        if "_cilindro_fantasma" in self.plotter.actors:
            self.plotter.remove_actor("_cilindro_fantasma")
            self.plotter.render()

    def stop_line_widget(self) -> None:
        self.ocultar_fantasma_cilindro()
        if self._line_widget is not None:
            try:
                self._line_widget.Off()
            except Exception:
                pass
            self._line_widget = None
        self.plotter.render()

    @staticmethod
    def _bounds_de(pts_c: np.ndarray) -> list[float]:
        """Bounds en el formato de PyVista, acotados al 50 % de la nube para que
        el widget arranque a una escala manejable en vez de abarcar el cerro."""
        mn, mx = pts_c.min(axis=0), pts_c.max(axis=0)
        c, half = (mn + mx) / 2, (mx - mn) * 0.25
        return [c[0] - half[0], c[0] + half[0],
                c[1] - half[1], c[1] + half[1],
                c[2] - half[2], c[2] + half[2]]

    # ------------------------------------------------------------------ #
    # Herramienta de lazo (VTK observer + vtkActor2D)                     #
    # ------------------------------------------------------------------ #

    def start_lasso(self, callback) -> None:
        """
        Activa el modo lazo. Los clicks sobre la nube agregan vertices;
        clic derecho cierra el poligono y llama a callback(verts).

        Usa observers VTK en lugar de un overlay Qt porque en Windows
        el widget VTK es una ventana nativa que captura todos los eventos
        de mouse antes de que Qt los entregue a widgets normales.
        """
        self._lasso_callback = callback
        self._lasso_screen_verts = []
        self._lasso_vtk_verts = []
        self._lasso_cursor_vtk = (0, 0)

        # Bloquear el estilo por defecto (rotacion, zoom, etc.)
        from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser
        vtk_iren = self.plotter.iren.interactor
        self._lasso_saved_style = vtk_iren.GetInteractorStyle()
        vtk_iren.SetInteractorStyle(vtkInteractorStyleUser())

        # Agregar observers para capturar eventos de mouse
        self._lasso_obs_ids = [
            vtk_iren.AddObserver("LeftButtonPressEvent",  self._vtk_lasso_click),
            vtk_iren.AddObserver("MouseMoveEvent",        self._vtk_lasso_move),
            vtk_iren.AddObserver("RightButtonPressEvent", self._vtk_lasso_close),
        ]

        # Crear actor 2D para dibujar el poligono en display coordinates
        import vtk
        self._lasso_pts2d  = vtk.vtkPoints()
        self._lasso_cells2d = vtk.vtkCellArray()
        self._lasso_poly2d  = vtk.vtkPolyData()
        self._lasso_poly2d.SetPoints(self._lasso_pts2d)
        self._lasso_poly2d.SetLines(self._lasso_cells2d)

        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToDisplay()

        mapper2d = vtk.vtkPolyDataMapper2D()
        mapper2d.SetInputData(self._lasso_poly2d)
        mapper2d.SetTransformCoordinate(coord)

        self._lasso_actor2d = vtk.vtkActor2D()
        self._lasso_actor2d.SetMapper(mapper2d)
        self._lasso_actor2d.GetProperty().SetColor(1, 1, 1)
        self._lasso_actor2d.GetProperty().SetLineWidth(2)

        self.plotter.renderer.AddActor2D(self._lasso_actor2d)
        self.plotter.render()

    # Radio (px) para cerrar el lazo clicando sobre el primer vertice
    LASSO_SNAP_PX = 15

    def _vtk_lasso_click(self, vtk_iren, event) -> None:
        x, y = vtk_iren.GetEventPosition()

        if len(self._lasso_vtk_verts) >= 3:
            # Doble clic (el adaptador Qt→VTK marca RepeatCount > 0)
            if vtk_iren.GetRepeatCount() > 0:
                self._vtk_lasso_close(vtk_iren, event)
                return
            # Clic sobre el primer vertice: unir inicio y final
            x0, y0 = self._lasso_vtk_verts[0]
            if abs(x - x0) + abs(y - y0) < self.LASSO_SNAP_PX:
                self._vtk_lasso_close(vtk_iren, event)
                return

        h = self.plotter.renderer.GetSize()[1]
        screen_y = h - y  # VTK: y=0 abajo → screen: y=0 arriba

        self._lasso_screen_verts.append((x, screen_y))
        self._lasso_vtk_verts.append((x, y))
        self._update_lasso_2d()

    def _vtk_lasso_move(self, vtk_iren, event) -> None:
        x, y = vtk_iren.GetEventPosition()
        self._lasso_cursor_vtk = (x, y)
        if self._lasso_vtk_verts:
            self._update_lasso_2d()

    def _vtk_lasso_close(self, vtk_iren, event) -> None:
        verts = self._lasso_screen_verts.copy()
        callback = self._lasso_callback
        # Restaurar de inmediato el control de cámara: el usuario debe poder
        # orbitar/zoomear para revisar el preview 3D antes de aplicar.
        self.stop_lasso()
        if callback:
            # Siempre notificar (incluso con <3 vertices) para que la ventana
            # principal restaure estados de UI; ella decide si el lazo es valido.
            callback(verts)

    def _update_lasso_2d(self) -> None:
        import vtk as _vtk

        self._lasso_pts2d.Reset()
        self._lasso_cells2d.Reset()

        vtk_verts = self._lasso_vtk_verts
        cx, cy = self._lasso_cursor_vtk
        n = len(vtk_verts)

        for x, y in vtk_verts:
            self._lasso_pts2d.InsertNextPoint(x, y, 0)

        # Aristas entre vertices consecutivos
        for i in range(n - 1):
            line = _vtk.vtkLine()
            line.GetPointIds().SetId(0, i)
            line.GetPointIds().SetId(1, i + 1)
            self._lasso_cells2d.InsertNextCell(line)

        # Linea de preview al cursor
        if n >= 1:
            self._lasso_pts2d.InsertNextPoint(cx, cy, 0)
            line = _vtk.vtkLine()
            line.GetPointIds().SetId(0, n - 1)
            line.GetPointIds().SetId(1, n)
            self._lasso_cells2d.InsertNextCell(line)

        # Borde de cierre: cursor → primer vertice, para ver el poligono unido
        if n >= 2:
            line = _vtk.vtkLine()
            line.GetPointIds().SetId(0, n)   # cursor (ultimo punto insertado)
            line.GetPointIds().SetId(1, 0)   # primer vertice
            self._lasso_cells2d.InsertNextCell(line)

        self._lasso_poly2d.Modified()
        self.plotter.render()

    def _cleanup_lasso_2d(self) -> None:
        if self._lasso_actor2d is not None:
            self.plotter.renderer.RemoveActor2D(self._lasso_actor2d)
            self._lasso_actor2d = None
        self._lasso_screen_verts = []
        self._lasso_vtk_verts = []
        self._lasso_cursor_vtk = (0, 0)

    def stop_lasso(self) -> None:
        # Remover observers VTK
        if self._lasso_obs_ids:
            vtk_iren = self.plotter.iren.interactor
            for obs_id in self._lasso_obs_ids:
                vtk_iren.RemoveObserver(obs_id)
            self._lasso_obs_ids = []

        # Restaurar estilo de interaccion
        if self._lasso_saved_style is not None:
            self.plotter.iren.interactor.SetInteractorStyle(self._lasso_saved_style)
            self._lasso_saved_style = None

        self._cleanup_lasso_2d()

        if "_lasso_selection" in self.plotter.actors:
            self.plotter.remove_actor("_lasso_selection")
        for name in ("cloud_lidar", "cloud_cropped"):
            actor = self.plotter.actors.get(name)
            if actor is not None:
                actor.GetProperty().SetOpacity(1.0)
        self.plotter.render()

    # ------------------------------------------------------------------ #
    # Proyeccion de nube a pantalla                                        #
    # ------------------------------------------------------------------ #

    def project_cloud_to_screen(
        self, pcd: o3d.geometry.PointCloud
    ) -> tuple[np.ndarray, np.ndarray]:
        renderer = self.plotter.renderer
        w, h = renderer.GetSize()
        cam = renderer.GetActiveCamera()

        pts = np.asarray(pcd.points)
        center = self._scene_center if self._scene_center is not None else pts.mean(axis=0)
        pts_c = (pts - center).astype(np.float64)

        def _mat4(m):
            return np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])

        V = _mat4(cam.GetViewTransformMatrix())
        aspect = w / h if h > 0 else 1.0
        P = _mat4(cam.GetProjectionTransformMatrix(aspect, -1, 1))

        n = len(pts_c)
        pts_h = np.hstack([pts_c, np.ones((n, 1))])
        clip = (P @ V @ pts_h.T).T

        clip_w = clip[:, 3]
        valid = clip_w > 0
        ndc = np.zeros((n, 3))
        ndc[valid] = clip[valid, :3] / clip_w[valid, np.newaxis]

        sx = (ndc[:, 0] + 1) * 0.5 * w
        # Mismo flip Y que _vtk_lasso_click: y=0 arriba para coincidir con screen_verts
        sy = (1 - (ndc[:, 1] + 1) * 0.5) * h
        return np.column_stack([sx, sy]), valid

    # ------------------------------------------------------------------ #
    # Previsualización verde/rojo y render por capas                       #
    # ------------------------------------------------------------------ #

    COLOR_KEEP = "#4caf50"     # verde: se conserva
    COLOR_DISCARD = "#f44336"  # rojo: se elimina

    def _add_points_actor(
        self, pts: np.ndarray, name: str, color: str, point_size: float
    ) -> None:
        center = self._scene_center if self._scene_center is not None else 0.0
        poly = pv.PolyData((pts - center).astype(np.float32))
        self.plotter.add_mesh(
            poly,
            color=color,
            point_size=point_size,
            render_points_as_spheres=True,
            style="points",
            name=name,
            reset_camera=False,   # el preview se redibuja al mover la caja: si
                                  # PyVista reencuadra, se pierde el punto de vista
        )

    def preview_split(
        self,
        pcd: o3d.geometry.PointCloud,
        keep_mask: np.ndarray,
        source_actor_name: str,
    ) -> None:
        """
        Previsualiza un recorte: VERDE = se conserva, ROJO = se elimina. Los
        actores fuente se atenúan para que dominen los colores.

        `source_actor_name` puede ser un nombre o una lista de nombres: el
        recorte se aplica a todas las capas visibles a la vez, no solo a la
        activa.
        """
        keep_mask = np.asarray(keep_mask, dtype=bool)
        pts = np.asarray(pcd.points)

        nombres = ([source_actor_name] if isinstance(source_actor_name, str)
                   else list(source_actor_name))
        self._dimmed_actor = []
        for nombre in nombres:
            actor = self.plotter.actors.get(nombre)
            if actor is not None:
                self._dimmed_actor.append(nombre)

        for name, sel in (("_preview_keep", keep_mask), ("_preview_discard", ~keep_mask)):
            if name in self.plotter.actors:
                self.plotter.remove_actor(name)
            if sel.any():
                color = self.COLOR_KEEP if name == "_preview_keep" else self.COLOR_DISCARD
                self._add_points_actor(pts[sel], name, color,
                                       point_size=self._tamano_punto)
        self._aplicar_estilo_nube()

    def mostrar_clusters(self, pcd, etiquetas, seleccion=(),
                         source_actor_name=None) -> None:
        """Colorea la nube por cluster: uno visible es uno distinguible.

        El ruido va en gris oscuro y lo seleccionado en verde, el mismo verde
        que el resto de la app usa para "esto es lo que se va a llevar". Los
        colores de cluster salen de un salto en el círculo cromático por la
        razón áurea, de modo que dos clusters con ids consecutivos —que suelen
        ser vecinos— quedan en colores bien distintos.
        """
        import colorsys

        etiquetas = np.asarray(etiquetas)
        pts = np.asarray(pcd.points)
        sel = set(int(s) for s in seleccion)

        nombres = ([] if source_actor_name is None else
                   [source_actor_name] if isinstance(source_actor_name, str)
                   else list(source_actor_name))
        self._dimmed_actor = [n for n in nombres if n in self.plotter.actors]

        colores = np.full((len(pts), 3), 0.25, dtype=np.float32)   # ruido
        for cid in np.unique(etiquetas):
            if cid == -1:
                continue
            m = etiquetas == cid
            if int(cid) in sel:
                colores[m] = (0.20, 0.90, 0.30)
            else:
                h = (int(cid) * 0.61803398875) % 1.0
                colores[m] = colorsys.hsv_to_rgb(h, 0.65, 0.95)

        centro = self._scene_center if self._scene_center is not None else 0.0
        poly = pv.PolyData((pts - centro).astype(np.float32))
        poly["RGB"] = (colores * 255).astype(np.uint8)
        self.plotter.add_mesh(poly, scalars="RGB", rgb=True,
                              point_size=self._tamano_punto,
                              render_points_as_spheres=True, style="points",
                              name="_clusters", reset_camera=False)
        self._aplicar_estilo_nube()

    def ocultar_clusters(self) -> None:
        if "_clusters" in self.plotter.actors:
            self.plotter.remove_actor("_clusters")
        self._dimmed_actor = None
        self._aplicar_estilo_nube()

    def clear_preview(self) -> None:
        """Quita los actores de previsualización y restaura la opacidad."""
        for name in self.ACTORES_PREVIEW:
            if name in self.plotter.actors:
                self.plotter.remove_actor(name)
        self._dimmed_actor = None
        self._aplicar_estilo_nube()

    def show_layers(self, stack) -> None:
        """
        Renderiza un actor por capa visible (layer_<i>). La capa activa a
        opacidad plena, las demás a 0.5. Reemplaza a los actores legacy.
        """
        # Quitar actores de capas anteriores y nubes legacy
        obsoletos = [
            n for n in list(self.plotter.actors)
            if n.startswith("layer_") or n in ("cloud_lidar", "cloud_cropped")
        ]
        for n in obsoletos:
            self.plotter.remove_actor(n)
            if n in self._active_actors:
                self._active_actors.remove(n)

        if self._scene_center is None and len(stack.layers) > 0:
            self._scene_center = np.asarray(stack.layers[0].pcd.points).mean(axis=0)

        for i, capa in enumerate(stack):
            if not capa.visible:
                continue
            name = f"layer_{i}"
            poly = _o3d_to_pyvista(capa.pcd, center=self._scene_center)
            kwargs = dict(
                point_size=self._tamano_punto,
                render_points_as_spheres=True,
                style="points",
                name=name,
                opacity=(self._opacidad_nube if i == stack.active_index
                         else self._opacidad_nube * 0.5),
                # solo se encuadra la primera vez: tras un recorte el usuario
                # quiere seguir mirando desde donde estaba
                reset_camera=not self._camara_encuadrada,
            )
            if capa.pcd.has_colors():
                self.plotter.add_mesh(poly, scalars="RGB", rgb=True, **kwargs)
            else:
                self.plotter.add_mesh(
                    poly, scalars="z", cmap="viridis", show_scalar_bar=False, **kwargs
                )
            self._active_actors.append(name)
        self._camara_encuadrada = True
        self._aplicar_estilo_nube()

    def set_actor_visibility(self, name: str, visible: bool) -> None:
        """Muestra u oculta un actor por nombre."""
        actor = self.plotter.actors.get(name)
        if actor is not None:
            actor.visibility = visible
            self.plotter.render()


# ------------------------------------------------------------------ #
# Funciones auxiliares de conversión                                   #
# ------------------------------------------------------------------ #

def _maybe_downsample(
    pcd: o3d.geometry.PointCloud,
    max_points: int,
):
    return pcd


def _o3d_to_pyvista(
    pcd: o3d.geometry.PointCloud,
    center: np.ndarray | None = None,
) -> pv.PolyData:
    """
    Convierte a PolyData restando `center` (o la media propia si es None).
    Todos los actores de una misma escena deben compartir el mismo centro
    para quedar alineados entre sí.
    """
    pts = np.asarray(
        pcd.points
    ).copy()

    if center is None:
        center = pts.mean(axis=0)

    print(
        "[viewer] centro original:",
        center
    )

    pts -= center

    poly = pv.PolyData(pts)

    if pcd.has_colors():

        colors = (
            np.asarray(pcd.colors) * 255
        ).astype(np.uint8)

        poly["RGB"] = colors

    else:

        poly["z"] = pts[:, 2]

    return poly


def _o3d_mesh_to_pyvista(mesh: o3d.geometry.TriangleMesh) -> pv.PolyData:
    """Convierte una malla Open3D a PolyData de PyVista."""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    # PyVista requiere [n_vertices, v0, v1, v2] por cada cara
    faces_pv = np.hstack([
        np.full((len(faces), 1), 3),
        faces
    ])
    return pv.PolyData(vertices, faces_pv)
