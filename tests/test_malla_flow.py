"""Flujo completo de mallado en la ventana real."""
import sys

import numpy as np
import open3d as o3d
import pytest
from PyQt6.QtWidgets import QApplication

from app.modules import meshing as M


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(sys.argv)


from tests.test_meshing import sin_cgal  # noqa: E402


def _ventana_con_esfera(qtbot, n=4000):
    from app.core.layers import CloudLayer, LayerStack
    from app.gui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    rng = np.random.default_rng(0)
    v = rng.normal(size=(n, 3))
    pts = v / np.linalg.norm(v, axis=1, keepdims=True)
    stack = LayerStack()
    stack.layers = [CloudLayer(
        name="Cúpula",
        pcd=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)))]
    stack.set_activas([0])
    w._layer_stack = stack
    return w, stack, pts


@sin_cgal
def test_flujo_completo_generar_y_exportar(app, qtbot, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    w, stack, pts = _ventana_con_esfera(qtbot)
    w._activate_tool("malla")
    assert w._active_tool == "malla"

    dock = w._ensure_crop_dock()
    # El limite se propone solo, a partir del espaciado de la entidad.
    assert "límite sugerido" in dock._lbl_ma_ref.text()
    assert dock._ma_sugerido == pytest.approx(M.perimetro_sugerido(pts), rel=1e-9)

    # Generar corre en un hilo: se lanza y se espera a que el panel lo refleje.
    dock._btn_ma_run.click()
    assert w._malla_worker is not None and dock._btn_ma_run.text() == "Generando…"
    with qtbot.waitSignal(w._malla_worker.finished, timeout=120_000):
        pass
    qtbot.waitUntil(lambda: w._malla is not None, timeout=10_000)

    assert w._malla is not None and w._malla.n_caras > 500
    assert dock._btn_ma_export.isEnabled()
    texto = dock._lbl_ma_res.text()
    assert "Triángulos" in texto and "Invención" in texto
    assert "_malla" in w.viewer.plotter.actors, "la malla debe dibujarse"

    # Exportar: el nombre por defecto sale del nombre de la entidad.
    destino = tmp_path / "cupula.ply"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(destino), ""))
    w._on_malla_exportar()
    assert destino.exists()
    leida = o3d.io.read_triangle_mesh(str(destino))
    assert len(leida.triangles) == w._malla.n_caras


@sin_cgal
def test_generar_sin_capas_en_uso_avisa_y_no_lanza_el_hilo(app, qtbot):
    w, stack, _ = _ventana_con_esfera(qtbot)
    w._activate_tool("malla")
    stack.set_activas([])
    w._malla_worker = None
    w._on_malla_generar(0.3, True, 5.0)
    assert w._malla_worker is None
    assert "Usar" in w._status_label.text()


@sin_cgal
def test_un_perimetro_imposible_reporta_el_error_sin_romper(app, qtbot):
    """Un limite ridiculamente pequeno no produce triangulos: debe avisar."""
    w, stack, _ = _ventana_con_esfera(qtbot)
    w._activate_tool("malla")
    dock = w._ensure_crop_dock()
    with qtbot.waitSignal(w._malla_worker.error, timeout=120_000) if False else \
            __import__("contextlib").nullcontext():
        w._on_malla_generar(0.001, False, 0.0)
        qtbot.waitUntil(lambda: not dock._btn_ma_run.text().startswith("Generando"),
                        timeout=120_000)
    assert w._malla is None
    assert "No se pudo generar" in dock._lbl_ma_res.text()


def test_la_malla_se_oculta_al_cambiar_de_herramienta(app, qtbot):
    w, stack, _ = _ventana_con_esfera(qtbot, n=500)
    w._activate_tool("malla")
    w.viewer.mostrar_malla(np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]]),
                           np.array([[0, 1, 2]], np.int32))
    assert "_malla" in w.viewer.plotter.actors
    w._activate_tool("esfera")
    assert "_malla" not in w.viewer.plotter.actors


@sin_cgal
def test_la_nube_no_debe_tapar_la_malla_recien_generada(app, qtbot):
    """Regresion reportada: la malla parecia no generarse.

    Advancing Front INTERPOLA los puntos, asi que la superficie pasa por cada
    uno de ellos. Con la nube dibujada encima, opaca, la malla queda tapada por
    completo y el resultado se ve identico a la nube de partida.
    """
    w, stack, _ = _ventana_con_esfera(qtbot, n=2000)
    w._activate_tool("malla")
    dock = w._ensure_crop_dock()

    dock._btn_ma_run.click()
    with qtbot.waitSignal(w._malla_worker.finished, timeout=120_000):
        pass
    qtbot.waitUntil(lambda: w._malla is not None, timeout=10_000)

    def capas_visibles():
        return [a.GetVisibility() for n, a in w.viewer.plotter.actors.items()
                if n.startswith("layer_")]

    # Por defecto se mira la malla sola: opaca y con las capas apagadas.
    assert dock.malla_vista() == "malla"
    assert "_malla" in w.viewer.plotter.actors
    assert all(v == 0 for v in capas_visibles()), "la nube taparia la malla"
    assert w.viewer.plotter.actors["_malla"].GetProperty().GetOpacity() == 1.0

    # 'Ambas': la malla se vuelve translucida y la nube reaparece.
    w._on_malla_vista("ambas")
    assert all(v == 1 for v in capas_visibles())
    assert w.viewer.plotter.actors["_malla"].GetProperty().GetOpacity() < 1.0

    # 'Nube': la malla desaparece y las capas vuelven.
    w._on_malla_vista("nube")
    assert "_malla" not in w.viewer.plotter.actors
    assert all(v == 1 for v in capas_visibles())


@sin_cgal
def test_salir_de_la_herramienta_devuelve_las_capas_a_la_vista(app, qtbot):
    """Apagar las capas es una ocultacion de vista, no un cambio de estado."""
    w, stack, _ = _ventana_con_esfera(qtbot, n=2000)
    w._activate_tool("malla")
    dock = w._ensure_crop_dock()
    dock._btn_ma_run.click()
    with qtbot.waitSignal(w._malla_worker.finished, timeout=120_000):
        pass
    qtbot.waitUntil(lambda: w._malla is not None, timeout=10_000)

    w._activate_tool("esfera")
    assert stack.layers[0].visible, "el estado de la capa no debio cambiar"
    assert all(a.GetVisibility() == 1
               for n, a in w.viewer.plotter.actors.items()
               if n.startswith("layer_"))


@sin_cgal
def test_el_panel_agrupa_las_metricas_por_lo_que_responden(app, qtbot):
    """Las cifras se leen mal en un parrafo corrido: cada una mide otra cosa."""
    w, stack, _ = _ventana_con_esfera(qtbot, n=2000)
    w._activate_tool("malla")
    dock = w._ensure_crop_dock()
    dock._btn_ma_run.click()
    with qtbot.waitSignal(w._malla_worker.finished, timeout=120_000):
        pass
    qtbot.waitUntil(lambda: w._malla is not None, timeout=10_000)

    html = dock._lbl_ma_res.text()
    for grupo in ("Malla", "Fidelidad", "Diagnóstico"):
        assert f"<b>{grupo}</b>" in html, grupo
    for etiqueta in ("Triángulos", "Vértices", "Cobertura", "resolución",
                     "Invención", "abiertos", "Cerrada"):
        assert etiqueta in html, etiqueta
    # Los vertices creados al parchear se separan de los medidos.
    assert "creados al parchear" in html


def test_las_metricas_no_dependen_de_haber_generado(app, qtbot):
    """Formateo puro: se puede comprobar sin CGAL."""
    from app.gui.panels.crop_dock import CropDock
    from app.modules.meshing import Fidelidad

    d = CropDock()
    qtbot.addWidget(d)
    fid = Fidelidad(cobertura_media=0.0084, cobertura_p95=0.0166,
                    invencion=0.0065, tolerancia=0.05, resolucion=0.0082)
    d.set_malla_metricas(181_841, 92_925, 88_577, fid,
                         {"n_holes": 18, "n_antes": 4_312, "mayor": 110.3,
                          "mayor_aristas": 2_307, "is_closed": False})
    html = d._lbl_ma_res.text()
    # Cuantos se cerraron es lo que dice si mover el umbral serviria de algo.
    assert "4,294" in html or "4294" in html, "cerrados = 4.312 - 18"
    # El perimetro mide recorrido, no tamano: las aristas lo aclaran.
    assert "110.3 m" in html and "2,307 aristas" in html
    assert "181,841" in html and "0.84 cm" in html and "0.65 %" in html
    assert "4,348" in html, "92.925 - 88.577 vertices creados al parchear"
    assert d._btn_ma_export.isEnabled() and d._fila_ma_vista.isEnabled()


def test_las_cabeceras_del_panel_van_en_blanco(app, qtbot):
    """Regresion de estilo: eran ambar y competian con las cifras."""
    from app.gui.panels.crop_dock import CropDock
    from app.modules.meshing import Fidelidad

    d = CropDock()
    qtbot.addWidget(d)
    fid = Fidelidad(cobertura_media=0.008, cobertura_p95=0.016,
                    invencion=0.008, tolerancia=0.05, resolucion=0.008)
    d.set_malla_metricas(100, 90, 88, fid, {"n_holes": 2, "is_closed": False})
    html = d._lbl_ma_res.text()
    assert "#ffb74d" not in html, "las cabeceras ya no son ambar"
    assert html.count("#ffffff") >= 3, "una cabecera blanca por grupo"


def test_mallar_tiene_boton_propio_fuera_de_segmentar(app, qtbot):
    """Mallar no extrae ninguna entidad: convierte una ya extraida."""
    from app.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    en_segmentar = [a.data() for a in w._btn_seg.menu().actions() if a.data()]
    assert "malla" not in en_segmentar
    assert w._btn_malla.defaultAction() is w._act_tool_malla

    # Se habilita con la nube, como Recortar y Segmentar.
    assert not w._btn_malla.isEnabled()
    w._habilitar_herramientas(True)
    assert w._btn_malla.isEnabled()


def test_la_ayuda_es_la_de_la_herramienta_activa(app, qtbot):
    """Regresion: mostraba siempre el texto del lazo."""
    from app.gui.panels.crop_dock import _AYUDA_TOOL, CropDock

    d = CropDock()
    qtbot.addWidget(d)
    # Cada herramienta tiene la suya, y ninguna queda sin cubrir.
    assert set(d.PAGINAS) == set(_AYUDA_TOOL)

    claves = {
        "esfera": "Recorte angular",
        "dbscan": "DESPUÉS de las primitivas",
        "simetria": "Concordancia",
        "malla": "Invención",
        "lazo": "proyección en pantalla",
    }
    for tool, marca in claves.items():
        d.set_tool(tool)
        assert d._tool_actual == tool
        assert marca in _AYUDA_TOOL[tool], f"{tool}: falta '{marca}'"
        # Y no arrastra el texto de otra herramienta.
        if tool != "lazo":
            assert "Iniciar lazo" not in _AYUDA_TOOL[tool], tool


def test_la_ayuda_esta_escrita_con_tildes(app, qtbot):
    """Regresion: se escribio sin acentos para evitar lios de codificacion, y el
    texto quedaba lleno de 'enganaria' y 'resolucion'."""
    from app.gui.panels.crop_dock import _AYUDA_TOOL, _COMUN

    todo = _COMUN + "".join(_AYUDA_TOOL.values())
    sin_tilde = [s for s in (
        "engana", "tamano", "resolucion", "Invencion", "ultimo", "triangulo",
        "vacios", "parametro", "Como leer", "proyeccion", "angulo", "faldon",
        "cascara", "Reconstruccion", "aproximacion", "semiangulo", "vegetacion",
        "DESPUES", "unico", "minimo", "fraccion", "cupula", "operacion",
    ) if s in todo]
    assert sin_tilde == [], sin_tilde


def test_la_ayuda_de_la_malla_explica_las_dos_direcciones(app, qtbot):
    """El texto denso no se entendia: las cifras se explican como preguntas."""
    from app.gui.panels.crop_dock import _AYUDA_TOOL

    ayuda = _AYUDA_TOOL["malla"]
    assert "&rarr;" not in ayuda and "->" not in ayuda, "sin flechas"
    # Cada cifra encabeza con el nombre que aparece en el panel, para poder
    # buscarla, y dice de entrada que pregunta responde.
    assert "<b>Cobertura.</b> Dice si <b>falta</b> superficie" in ayuda
    assert "<b>Invención.</b> Dice si <b>sobra</b> superficie" in ayuda
    assert "<b>Contornos.</b> Dicen qué quedó <b>abierto</b>" in ayuda
    # Con un ejemplo numerico, que es lo que hace entender la resolucion.
    assert "0,84 cm" in ayuda and "0,82 cm" in ayuda


def test_ningun_texto_de_interfaz_lleva_raya(app, qtbot):
    """Regresion de estilo: fuera las rayas del texto que ve el usuario.

    Recorre las cadenas literales de todo el paquete y excluye las que son
    documentacion (docstrings de modulo, clase, funcion y de constante), que no
    llegan a la pantalla.
    """
    import ast
    from pathlib import Path

    RAYA = chr(8212)
    culpables = []
    for fuente in Path("app").rglob("*.py"):
        arbol = ast.parse(fuente.read_text(encoding="utf-8"))
        docs = set()
        for nodo in ast.walk(arbol):
            cuerpo = getattr(nodo, "body", None)
            if isinstance(cuerpo, list):
                # Docstrings de modulo/clase/funcion y de atributo: toda cadena
                # suelta como sentencia.
                for sent in cuerpo:
                    if (isinstance(sent, ast.Expr)
                            and isinstance(sent.value, ast.Constant)
                            and isinstance(sent.value.value, str)):
                        docs.add(id(sent.value))
        for nodo in ast.walk(arbol):
            if (isinstance(nodo, ast.Constant) and isinstance(nodo.value, str)
                    and RAYA in nodo.value and id(nodo) not in docs):
                culpables.append(f"{fuente}:{nodo.lineno}")
    assert culpables == [], culpables
