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

from PyQt6.QtCore import pyqtSignal


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
        # Centro compartido de la escena: todos los actores (capas, previews)
        # se dibujan restando este centro para quedar alineados.
        self._scene_center: np.ndarray | None = None
        self._dimmed_actor: list[str] | None = None   # actores atenuados por el preview
        self._camara_encuadrada = False   # se encuadra una vez, al cargar
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
                actor.GetProperty().SetOpacity(0.15)
                self._dimmed_actor.append(nombre)

        for name, sel in (("_preview_keep", keep_mask), ("_preview_discard", ~keep_mask)):
            if name in self.plotter.actors:
                self.plotter.remove_actor(name)
            if sel.any():
                color = self.COLOR_KEEP if name == "_preview_keep" else self.COLOR_DISCARD
                self._add_points_actor(pts[sel], name, color, point_size=4.0)
        self.plotter.render()

    def clear_preview(self) -> None:
        """Quita los actores de previsualización y restaura la opacidad."""
        for name in ("_preview_keep", "_preview_discard", "_lasso_selection"):
            if name in self.plotter.actors:
                self.plotter.remove_actor(name)
        for nombre in (self._dimmed_actor or []):
            actor = self.plotter.actors.get(nombre)
            if actor is not None:
                actor.GetProperty().SetOpacity(1.0)
        self._dimmed_actor = None
        self.plotter.render()

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
                point_size=2.0,
                render_points_as_spheres=True,
                style="points",
                name=name,
                opacity=1.0 if i == stack.active_index else 0.5,
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
        self.plotter.render()

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
