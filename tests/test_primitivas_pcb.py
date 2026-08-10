"""tests/test_primitivas_pcb.py — Plano, cilindro y cono.

Se usan superficies sintéticas con parámetros conocidos: si el ajuste recupera
el parámetro real dentro de la tolerancia del ruido, funciona. Los tests de
recorte (radio, tramo de eje, sector angular) documentan por qué esos controles
no son cosméticos: sin ellos la primitiva absorbe elementos vecinos.
"""
import numpy as np
import pytest

from app.modules.primitivas import (
    Cilindro, Plano,
    ajustar_cilindro, ajustar_plano,
    mascara_cilindro, mascara_plano,
)


# ---------------------------------------------------------------- Plano ---- #

def _nube_plana(n=4000, z=2.0, ruido=0.01, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-5, 5, (n, 2))
    return np.column_stack([xy, np.full(n, z) + rng.normal(0, ruido, n)])


def test_ajustar_plano_recupera_el_plano_real():
    pts = _nube_plana(z=2.0)
    p = ajustar_plano(pts, [0, 0, 2.3], [0, 0, 1], radio=6.0)
    # el plano z=2 es normal·x + d = 0 con normal=(0,0,1), d=-2
    assert abs(abs(p.normal[2]) - 1.0) < 1e-3
    assert abs(p.d + 2.0) < 0.05


def test_ajustar_plano_conserva_la_orientacion_del_usuario():
    """RANSAC devuelve la normal con signo arbitrario; si no se alinea con la
    que puso el usuario, el filtro por lado queda invertido."""
    pts = _nube_plana(z=2.0)
    arriba = ajustar_plano(pts, [0, 0, 2.3], [0, 0, 1], radio=6.0)
    abajo = ajustar_plano(pts, [0, 0, 2.3], [0, 0, -1], radio=6.0)
    assert arriba.normal[2] > 0 and abajo.normal[2] < 0


def test_el_radio_del_plano_evita_capturar_todo_el_terreno():
    """Un plano es infinito: sin acotarlo se lleva cualquier superficie
    coplanar del otro extremo de la nube."""
    cerca = _nube_plana(n=2000, z=2.0, seed=1)
    lejos = cerca + np.array([100.0, 0.0, 0.0])     # mismo plano, 100 m allá
    pts = np.vstack([cerca, lejos])
    plano = Plano(normal=[0, 0, 1], d=-2.0, centro=[0, 0, 2], radio=8.0)

    acotado = mascara_plano(pts, plano, tolerancia=0.08)
    infinito = mascara_plano(pts, Plano(normal=[0, 0, 1], d=-2.0), tolerancia=0.08)
    assert acotado.sum() == pytest.approx(len(cerca), rel=0.02)
    assert infinito.sum() > 1.9 * len(cerca)        # se llevó las dos


def test_el_lado_del_plano_separa_las_dos_caras_de_un_muro():
    """Con tolerancia simétrica las dos caras de un elemento plano delgado caen
    juntas. Aplica a muros y paneles, NO a la cúpula (que es esférica y se
    captura con la esfera). Cuando se conoce la procedencia de scan, separar por
    ahí es mejor criterio que por lado del plano."""
    rng = np.random.default_rng(2)
    xy = rng.uniform(-2, 2, (1500, 2))
    interior = np.column_stack([xy, np.full(1500, 1.97)])
    exterior = np.column_stack([xy, np.full(1500, 2.03)])
    pts = np.vstack([interior, exterior])
    plano = Plano(normal=[0, 0, 1], d=-2.0, centro=[0, 0, 2], radio=5.0)

    assert mascara_plano(pts, plano, 0.08, lado="ambos").sum() == 3000
    assert mascara_plano(pts, plano, 0.08, lado="positivo").sum() == 1500
    assert mascara_plano(pts, plano, 0.08, lado="negativo").sum() == 1500


def test_el_disco_del_plano_no_se_deforma_al_inclinarlo():
    """La distancia se mide dentro del plano, no en el espacio: si se midiera en
    el espacio, el disco se volvería un elipsoide al inclinar el plano."""
    plano = Plano(normal=[1, 1, 0], d=0.0, centro=[0, 0, 0], radio=3.0)
    sobre_el_plano = np.array([[0.0, 0.0, 2.9], [0.0, 0.0, 3.1]])
    m = mascara_plano(sobre_el_plano, plano, tolerancia=0.08)
    assert m[0] and not m[1]


def test_ajustar_plano_avisa_si_esta_lejos_de_los_datos():
    pts = _nube_plana(z=2.0)
    with pytest.raises(RuntimeError, match="cerca del plano"):
        ajustar_plano(pts, [0, 0, 50.0], [0, 0, 1], radio=6.0)


# ------------------------------------------------------------- Cilindro ---- #

def _nube_cilindrica(n=6000, radio=3.0, h=(0.0, 5.0), ruido=0.01,
                     centro=(0.0, 0.0), seed=0):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n)
    r = radio + rng.normal(0, ruido, n)
    return np.column_stack([centro[0] + r * np.cos(th),
                            centro[1] + r * np.sin(th),
                            rng.uniform(*h, n)])


def test_ajustar_cilindro_recupera_radio_y_eje():
    pts = _nube_cilindrica(radio=3.0, centro=(1.0, -2.0))
    c = ajustar_cilindro(pts, [1.3, -1.7, 0.0], [0, 0, 1], radio_aprox=3.2)
    assert abs(c.radio - 3.0) < 0.05
    assert np.allclose(c.punto[:2], [1.0, -2.0], atol=0.05)


def test_el_tramo_de_eje_evita_absorber_lo_que_esta_alineado_arriba():
    """Un cilindro es infinito a lo largo del eje: sin h_min/h_max, capturar el
    tambor se llevaría cualquier elemento del mismo radio más arriba."""
    tambor = _nube_cilindrica(n=3000, radio=3.0, h=(0.0, 4.0), seed=3)
    antena = _nube_cilindrica(n=1000, radio=3.0, h=(9.0, 11.0), seed=4)
    pts = np.vstack([tambor, antena])
    cil = Cilindro(punto=[0, 0, 0], eje=[0, 0, 1], radio=3.0)

    assert mascara_cilindro(pts, cil, 0.08).sum() == len(pts)
    acotado = mascara_cilindro(pts, cil, 0.08, h_min=-0.5, h_max=5.0)
    assert acotado.sum() == pytest.approx(len(tambor), rel=0.02)


def test_el_sector_angular_del_cilindro_recorta_media_cascara():
    pts = _nube_cilindrica(n=4000, radio=3.0, seed=5)
    cil = Cilindro(punto=[0, 0, 0], eje=[0, 0, 1], radio=3.0)
    media = mascara_cilindro(pts, cil, 0.08, theta_min=0.0, theta_max=180.0)
    assert media.sum() == pytest.approx(len(pts) / 2, rel=0.10)
    assert np.all(pts[media][:, 1] > -0.2)      # el semiplano y >= 0


def test_el_angulo_cero_del_cilindro_vertical_apunta_al_eje_x():
    """El usuario mueve un slider de ángulo: el origen tiene que ser predecible,
    no una base ortonormal cualquiera."""
    cil = Cilindro(punto=[0, 0, 0], eje=[0, 0, 1], radio=1.0)
    u, v = cil.base()
    assert np.allclose(u, [1, 0, 0], atol=1e-9)
    assert np.allclose(v, [0, 1, 0], atol=1e-9)
    _, _, th = cil.coords_locales(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    assert th[0] == pytest.approx(0.0) and th[1] == pytest.approx(90.0)


def test_el_sector_angular_puede_cruzar_el_cero():
    """De 350 a 10 grados es un sector de 20, no de 340: es lo que el usuario
    espera al arrastrar el sector por encima del origen."""
    pts = _nube_cilindrica(n=6000, radio=3.0, seed=6)
    cil = Cilindro(punto=[0, 0, 0], eje=[0, 0, 1], radio=3.0)
    m = mascara_cilindro(pts, cil, 0.08, theta_min=350.0, theta_max=10.0)
    assert m.sum() == pytest.approx(len(pts) * 20 / 360, rel=0.30)
    assert m.sum() < len(pts) / 2


def test_la_base_del_cilindro_es_ortonormal_para_cualquier_eje():
    for eje in ([0, 0, 1], [1, 0, 0], [1, 1, 1], [0.3, -0.9, 0.2]):
        u, v = Cilindro(eje=eje).base()
        e = Cilindro(eje=eje).eje
        assert np.allclose([u @ v, u @ e, v @ e], 0.0, atol=1e-9)
        assert np.allclose([np.linalg.norm(u), np.linalg.norm(v)], 1.0)


def test_ajustar_cilindro_funciona_con_eje_inclinado():
    """El eje lo pone el usuario y no se refina, pero el ajuste del radio debe
    funcionar igual para un eje cualquiera."""
    rng = np.random.default_rng(7)
    eje = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    cil = Cilindro(punto=[0, 0, 0], eje=eje, radio=2.0)
    u, v = cil.base()
    th, h = rng.uniform(0, 2 * np.pi, 5000), rng.uniform(0, 6, 5000)
    r = 2.0 + rng.normal(0, 0.01, 5000)
    pts = (r * np.cos(th))[:, None] * u + (r * np.sin(th))[:, None] * v + h[:, None] * eje

    ajustado = ajustar_cilindro(pts, [0, 0, 0], eje, radio_aprox=2.2)
    assert abs(ajustado.radio - 2.0) < 0.05


# ----------------------------------------------------------------- Cono ---- #

def _nube_conica(n=8000, r0=1.0, pend=0.5, h=(0.0, 4.0), ruido=0.01, seed=0):
    """Cono de radio r0 en h=0 que crece `pend` metros de radio por metro."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(*h, n)
    r = r0 + pend * z + rng.normal(0, ruido, n)
    return np.column_stack([r * np.cos(th), r * np.sin(th), z])


def test_ajustar_cono_recupera_radio_y_pendiente():
    from app.modules.primitivas import ajustar_cono
    pts = _nube_conica(r0=1.0, pend=0.5)
    c = ajustar_cono(pts, [0, 0, 0], [0, 0, 1], radio_aprox=2.0,
                     h_min=0, h_max=4, eps=0.03)
    assert abs(c.r0 - 1.0) < 0.05
    assert abs(c.pendiente - 0.5) < 0.02
    assert abs(c.semiangulo - np.degrees(np.arctan(0.5))) < 1.5


def test_el_cono_con_pendiente_cero_es_un_cilindro():
    from app.modules.primitivas import ajustar_cono
    pts = _nube_cilindrica(radio=3.0, h=(0.0, 4.0), seed=9)
    c = ajustar_cono(pts, [0, 0, 0], [0, 0, 1], radio_aprox=3.0,
                     h_min=0, h_max=4, eps=0.03)
    assert abs(c.pendiente) < 0.02
    assert c.vertice() is None or abs(c.r0 - 3.0) < 0.05


def test_el_vertice_del_cono_es_donde_el_radio_se_anula():
    from app.modules.primitivas import Cono
    c = Cono(punto=[0, 0, 0], eje=[0, 0, 1], r0=2.0, pendiente=0.5)
    v = c.vertice()
    assert v is not None and np.allclose(v, [0, 0, -4.0])   # 2 - 0.5*4 = 0


def test_la_tolerancia_del_cono_se_mide_perpendicular_a_la_superficie():
    """En un cono inclinado la diferencia de radios sobreestima la distancia
    real; si no se corrige, la tolerancia deja de significar metros."""
    from app.modules.primitivas import Cono, mascara_cono
    c = Cono(punto=[0, 0, 0], eje=[0, 0, 1], r0=1.0, pendiente=1.0)  # 45°
    # punto con exceso de radio 0.10 -> distancia real 0.10/sqrt(2) = 0.071
    p = np.array([[1.10, 0.0, 0.0]])
    assert mascara_cono(p, c, tolerancia=0.08)[0]        # 0.071 < 0.08
    assert not mascara_cono(p, c, tolerancia=0.06)[0]


def test_un_cono_captura_el_faldon_mejor_que_una_esfera():
    """Motivación de la primitiva: una superficie cónica ajustada con esfera
    obliga a una tolerancia enorme, que es lo que absorbe elementos vecinos."""
    from app.modules.primitivas import (Cono, ajustar_cono, ajustar_esfera,
                                        mascara_cono, mascara_esfera)
    pts = _nube_conica(n=6000, r0=1.0, pend=0.6, h=(0.0, 3.0), seed=11)
    cono = ajustar_cono(pts, [0, 0, 0], [0, 0, 1], radio_aprox=2.0,
                        h_min=0, h_max=3, eps=0.03)
    esfera = ajustar_esfera(pts, [0, 0, -3.0], 4.0, eps=0.05)

    con_cono = mascara_cono(pts, cono, tolerancia=0.05, h_min=0, h_max=3).sum()
    con_esfera = mascara_esfera(pts, esfera, tolerancia=0.05).sum()
    assert con_cono > 0.95 * len(pts)      # el cono captura casi todo
    assert con_cono > 2 * con_esfera       # la esfera se queda muy corta


def test_cambiar_la_apertura_gira_en_torno_a_la_mitad_del_tramo():
    """Si se cambiara solo la pendiente, al abrir el cono la superficie se
    despegaría de los datos y habría que recolocarlo entero."""
    from app.modules.primitivas import Cono, con_apertura
    base = Cono(punto=[0, 0, 0], eje=[0, 0, 1], r0=1.0, pendiente=0.0)
    h_min, h_max = 0.0, 4.0
    r_medio = float(base.radio_en(2.0))

    for grados in (10.0, 30.0, -25.0):
        c = con_apertura(base, grados, h_min, h_max)
        assert c.semiangulo == pytest.approx(abs(grados), abs=1e-6)
        # el radio en la mitad NO cambia: el cono sigue apoyado donde estaba
        assert float(c.radio_en(2.0)) == pytest.approx(r_medio, abs=1e-9)
        assert np.sign(c.pendiente) == np.sign(grados)


def test_la_apertura_cero_devuelve_un_cilindro():
    from app.modules.primitivas import Cono, con_apertura
    c = con_apertura(Cono(r0=2.0, pendiente=0.7), 0.0, 0.0, 3.0)
    assert c.pendiente == pytest.approx(0.0)
    assert c.vertice() is None


def test_la_apertura_necesita_un_tramo_finito():
    """Sin tramo no hay 'mitad' sobre la que girar."""
    from app.modules.primitivas import Cono, con_apertura
    with pytest.raises(ValueError):
        con_apertura(Cono(), 20.0, -np.inf, np.inf)
