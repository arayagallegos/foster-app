"""tests/test_meshing.py — Reconstruccion de malla por entidad.

Las funciones geometricas (espaciado, muestreo por area, fidelidad) se prueban
sin CGAL, con mallas construidas a mano de las que se conoce la respuesta. Las
que llaman al binding se saltan si no esta compilado.
"""
import numpy as np
import pytest

from app.modules import meshing as M


def _cgal_disponible() -> bool:
    try:
        M._cgal()
        return True
    except Exception:
        return False


sin_cgal = pytest.mark.skipif(
    not _cgal_disponible(), reason="binding de CGAL no compilado")


def _cuadrado(lado=1.0) -> M.Malla:
    """Dos triangulos que forman un cuadrado en z=0. Area conocida: lado^2."""
    v = np.array([[0, 0, 0], [lado, 0, 0], [lado, lado, 0], [0, lado, 0]], float)
    f = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    return M.Malla(v, f)


# ------------------------------------------------------------------ espaciado

def test_espaciado_de_una_rejilla_regular():
    """En una rejilla de paso 0,1 el vecino mas cercano esta a 0,1."""
    g = np.arange(10) * 0.1
    x, y = np.meshgrid(g, g)
    pts = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    assert M.espaciado_medio(pts) == pytest.approx(0.1, abs=1e-9)


def test_el_perimetro_sugerido_es_la_proporcion_documentada():
    g = np.arange(10) * 0.1
    x, y = np.meshgrid(g, g)
    pts = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    assert M.perimetro_sugerido(pts) == pytest.approx(14.0 * 0.1, rel=1e-9)


def test_espaciado_exige_al_menos_dos_puntos():
    with pytest.raises(ValueError):
        M.espaciado_medio(np.zeros((1, 3)))


# ------------------------------------------------------------------ la malla

def test_area_de_la_malla():
    assert _cuadrado(2.0).area == pytest.approx(4.0)
    assert _cuadrado().n_caras == 2 and _cuadrado().n_vertices == 4


def test_el_muestreo_cae_sobre_la_superficie():
    m = _cuadrado()
    s = M.muestrear_superficie(m, 5_000)
    assert len(s) == 5_000
    assert np.allclose(s[:, 2], 0.0)                       # el plano z=0
    assert s[:, :2].min() >= -1e-9 and s[:, :2].max() <= 1 + 1e-9


def test_el_muestreo_es_uniforme_por_area_no_por_triangulo():
    """Un triangulo 99 veces mayor debe recibir ~99 veces mas muestras.

    Si se eligiera el triangulo al azar con igual probabilidad, la mitad de las
    muestras caeria en el diminuto y la medida de invencion quedaria dominada
    por el detalle en vez de por la superficie.
    """
    v = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0],       # area 50
                  [0, 0, 1], [0.1, 0, 1], [0, 0.1, 1]], float)  # area 0,005
    m = M.Malla(v, np.array([[0, 1, 2], [3, 4, 5]], np.int32))
    s = M.muestrear_superficie(m, 20_000)
    en_el_grande = (s[:, 2] < 0.5).mean()
    assert en_el_grande > 0.99


def test_muestrear_una_malla_vacia_no_revienta():
    m = M.Malla(np.zeros((0, 3)), np.zeros((0, 3), np.int32))
    assert len(M.muestrear_superficie(m, 100)) == 0


# ---------------------------------------------------------------- fidelidad

def test_fidelidad_perfecta_cuando_la_malla_pasa_por_los_puntos():
    """Sin invencion, y cobertura por debajo del suelo del propio muestreo.

    No puede dar cero exacto: medir contra una superficie muestreada deja a
    cualquier punto a media separacion de muestras del vecino mas proximo. Por
    eso `Fidelidad` reporta esa resolucion junto a la cobertura.
    """
    m = _cuadrado()
    # Puntos densos SOBRE la malla. Las cuatro esquinas no bastan: dejarian el
    # 99 % de la superficie a mas de 5 cm de todo punto real, y con razon.
    g = np.linspace(0, 1, 60)
    x, y = np.meshgrid(g, g)
    pts = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])

    fid = M.fidelidad(m, pts, n_muestras=5_000)
    assert fid.invencion == pytest.approx(0.0, abs=1e-12)
    assert fid.cobertura_media <= 2 * fid.resolucion, fid.resumen()
    assert fid.resolucion == pytest.approx(0.5 * (1.0 / 5_000) ** 0.5, rel=1e-6)


def test_mas_muestras_afinan_la_resolucion():
    m = _cuadrado()
    g = np.linspace(0, 1, 60)
    x, y = np.meshgrid(g, g)
    pts = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    gruesa = M.fidelidad(m, pts, n_muestras=1_000)
    fina = M.fidelidad(m, pts, n_muestras=100_000)
    assert fina.resolucion < gruesa.resolucion / 5
    assert fina.cobertura_media < gruesa.cobertura_media


def test_la_invencion_detecta_superficie_sin_datos():
    """La mitad de la malla no tiene ningun punto real debajo."""
    m = _cuadrado(2.0)
    rng = np.random.default_rng(0)
    # Puntos densos solo en la mitad x < 1.
    pts = np.column_stack([rng.uniform(0, 1, 4000), rng.uniform(0, 2, 4000),
                           np.zeros(4000)])
    fid = M.fidelidad(m, pts, tolerancia=0.05, n_muestras=40_000)
    assert 0.4 < fid.invencion < 0.6, fid.resumen()


def test_la_cobertura_detecta_superficie_que_falta():
    """Puntos reales lejos de la malla: la cobertura los delata."""
    m = _cuadrado()
    pts = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.30]])     # uno a 30 cm
    fid = M.fidelidad(m, pts, n_muestras=1_000)
    assert fid.cobertura_media > 0.05


def test_fidelidad_sin_puntos_es_un_error():
    with pytest.raises(ValueError):
        M.fidelidad(_cuadrado(), np.zeros((0, 3)))


def test_las_dos_direcciones_son_independientes():
    """El caso que motivo la metrica bidireccional: cobertura excelente
    conviviendo con superficie fabricada."""
    m = _cuadrado(2.0)
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(0, 1, 3000), rng.uniform(0, 2, 3000),
                           np.zeros(3000)])
    fid = M.fidelidad(m, pts, n_muestras=30_000)
    assert fid.cobertura_media < 0.02, "los puntos SI estan sobre la malla"
    assert fid.invencion > 0.4, "y aun asi media malla es inventada"


# ------------------------------------------------------- con CGAL de verdad

@sin_cgal
def test_reconstruir_una_esfera_produce_malla_manifold_y_fiel():
    """Caso completo sobre geometria conocida."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=(6000, 3))
    pts = v / np.linalg.norm(v, axis=1, keepdims=True)      # esfera de radio 1

    # Sin rellenar, Advancing Front INTERPOLA: los vertices son los puntos.
    cruda = M.reconstruir(pts, rellenar=False)
    assert cruda.n_vertices == pytest.approx(len(pts), rel=0.02)

    malla = M.reconstruir(pts)
    assert malla.n_caras > 1000
    # Al cerrar agujeros se anaden vertices en los parches, nunca se quitan.
    assert malla.n_vertices >= cruda.n_vertices
    # Area de la esfera unidad: 4*pi ~ 12,57
    assert malla.area == pytest.approx(4 * np.pi, rel=0.1)

    fid = M.fidelidad(malla, pts)
    # La malla interpola los puntos: la cobertura debe quedar en el suelo del
    # muestreo, no en un valor que indique superficie faltante.
    assert fid.cobertura_media < 3 * fid.resolucion, fid.resumen()
    assert fid.cobertura_media < 0.01, fid.resumen()
    assert fid.invencion < 0.05, fid.resumen()

    st = M.diagnostico(malla)
    assert st["self_intersects"] is False or st["self_intersects"] == 0


@sin_cgal
def test_el_limite_de_perimetro_evita_puentear_un_vacio():
    """Con una franja sin datos, un limite generoso la puentea y uno ajustado no.

    Es el argumento que decidio el metodo: sin limite, el frente de avance
    fabrica superficie sobre el vacio.
    """
    rng = np.random.default_rng(0)
    p = rng.uniform(-1, 1, (12000, 2))
    p = p[np.abs(p[:, 0]) > 0.35]                # franja vacia de 0,7 m de ancho
    pts = np.column_stack([p, rng.normal(0, 0.002, len(p))])

    apretado = M.reconstruir(pts, max_perimetro=M.perimetro_sugerido(pts),
                             rellenar=False)
    suelto = M.reconstruir(pts, max_perimetro=5.0, rellenar=False)
    assert suelto.area > apretado.area * 1.2, (
        f"suelto={suelto.area:.3f} apretado={apretado.area:.3f}")

    f_ap = M.fidelidad(apretado, pts)
    f_su = M.fidelidad(suelto, pts)
    assert f_ap.invencion < f_su.invencion


@sin_cgal
def test_reconstruir_exige_puntos_suficientes():
    with pytest.raises(ValueError):
        M.reconstruir(np.zeros((3, 3)))


@sin_cgal
def test_exportar_escribe_un_ply_legible(tmp_path):
    import open3d as o3d

    rng = np.random.default_rng(0)
    v = rng.normal(size=(3000, 3))
    pts = v / np.linalg.norm(v, axis=1, keepdims=True)
    malla = M.reconstruir(pts)

    destino = tmp_path / "sub" / "cupula.ply"
    M.exportar(malla, destino)
    assert destino.exists()
    leida = o3d.io.read_triangle_mesh(str(destino))
    assert len(leida.triangles) == malla.n_caras
    assert leida.has_vertex_normals()


@sin_cgal
def test_un_agujero_es_un_contorno_no_un_hueco_visual():
    """El borde exterior cuenta como UNO por dentado que se vea.

    Es la causa de que el recuento parezca no corresponder con la pantalla: un
    parche con el borde ondulado en treinta entrantes y tres agujeros interiores
    reporta el contorno exterior como un unico contorno de cientos de aristas,
    no como treinta agujeros.
    """
    rng = np.random.default_rng(0)
    p = rng.uniform(-1, 1, (30_000, 2))
    r = np.linalg.norm(p, axis=1)
    th = np.arctan2(p[:, 1], p[:, 0])
    p = p[r < 0.9 + 0.08 * np.sin(30 * th)]        # borde con 30 entrantes
    for cx, cy in [(-0.4, 0.3), (0.35, -0.2), (0.1, 0.5)]:
        p = p[np.linalg.norm(p - [cx, cy], axis=1) > 0.12]   # 3 agujeros
    pts = np.column_stack([p, rng.normal(0, 0.001, len(p))])

    m = M.reconstruir(pts, rellenar=False)
    huecos = M.agujeros(m)
    assert huecos == sorted(huecos, key=lambda h: -h["perimetro"]), "de mayor a menor"

    # El contorno exterior: uno solo, con cientos de aristas y perimetro enorme.
    exterior = huecos[0]
    assert exterior["aristas"] > 300
    assert exterior["perimetro"] > 5.0
    # Los tres interiores, muy por detras en tamano y parecidos entre si.
    interiores = huecos[1:4]
    assert all(0.5 < h["perimetro"] < 1.5 for h in interiores), interiores


@sin_cgal
def test_rellenar_cierra_los_pequenos_y_respeta_los_grandes():
    """El umbral es lo que separa 'oclusion' de 'zona sin cobertura'."""
    rng = np.random.default_rng(0)
    p = rng.uniform(-1, 1, (20_000, 2))
    p = p[np.linalg.norm(p, axis=1) < 0.95]
    p = p[np.linalg.norm(p - [0.3, 0.0], axis=1) > 0.30]     # un agujero grande
    p = p[np.linalg.norm(p + [0.4, 0.3], axis=1) > 0.06]     # y uno pequeno
    pts = np.column_stack([p, rng.normal(0, 0.001, len(p))])

    cruda = M.reconstruir(pts, rellenar=False)
    antes = M.agujeros(cruda)
    # Umbral entre ambos: cierra el pequeno y deja el grande.
    rellena = M.rellenar_agujeros(cruda, 1.0)
    despues = M.agujeros(rellena)
    assert len(despues) < len(antes)
    assert any(h["perimetro"] > 1.0 for h in despues), "el grande sigue abierto"
    assert rellena.area > cruda.area, "cerrar agujeros fabrica superficie"
