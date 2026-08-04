"""tests/test_primitivas.py — Motor de segmentación por primitivas."""
import numpy as np
import pytest

from tests.synthetic_cloud import _dome, _ring


def _nube_cupula(seed=0, r=2.5, cz=3.0, n=6000, ruido=0.01):
    """Semiesfera sobre un anillo: imita cúpula + tambor."""
    rng = np.random.default_rng(seed)
    cup = _dome(r, cz, n, rng)
    tam = _ring(r, 0.0, cz, n // 2, rng)
    pts = np.vstack([cup, tam]) + rng.normal(0, ruido, (n + n // 2, 3))
    verdad = np.concatenate([np.ones(n, bool), np.zeros(n // 2, bool)])
    return pts, verdad


def test_ajustar_esfera_corrige_una_colocacion_aproximada():
    from app.modules.primitivas import ajustar_esfera
    pts, _ = _nube_cupula()
    # el usuario coloca la esfera descentrada y con radio equivocado
    esf = ajustar_esfera(pts, centro_aprox=(0.4, -0.3, 3.4), radio_aprox=2.2)
    assert esf.radio == pytest.approx(2.5, abs=0.05)
    assert esf.centro == pytest.approx([0.0, 0.0, 3.0], abs=0.05)


def test_mascara_captura_la_cupula():
    from app.modules.primitivas import ajustar_esfera, mascara_esfera
    pts, verdad = _nube_cupula()
    esf = ajustar_esfera(pts, (0.0, 0.0, 3.0), 2.5)
    m = mascara_esfera(pts, esf, tolerancia=0.05)
    assert np.mean(m[verdad]) > 0.95          # casi toda la cúpula


def test_el_recorte_angular_evita_absorber_el_tambor():
    """El borde superior del tambor coincide geométricamente con el ecuador de
    la esfera: esos puntos SÍ están sobre la cáscara, así que ninguna tolerancia
    los excluye. Solo el recorte angular los deja fuera.

    Es el mismo fenómeno medido en el Observatorio, donde las compuertas están
    sobre la esfera de la cúpula pero en otro rango de phi.
    """
    from app.modules.primitivas import ajustar_esfera, mascara_esfera
    pts, verdad = _nube_cupula()
    esf = ajustar_esfera(pts, (0.0, 0.0, 3.0), 2.5)

    sin_recorte = mascara_esfera(pts, esf, tolerancia=0.05)
    con_recorte = mascara_esfera(pts, esf, tolerancia=0.05, phi_max=85.0)

    # sin recorte se cuela una parte del tambor (la que toca el ecuador)
    assert np.mean(sin_recorte[~verdad]) > 0.15
    # con recorte, prácticamente nada
    assert np.mean(con_recorte[~verdad]) < 0.02
    # y la cúpula sigue bien capturada
    assert np.mean(con_recorte[verdad]) > 0.90


def test_recorte_angular_descarta_bajo_el_ecuador():
    """phi = 0 en la cúspide, 90 en el ecuador. Recortar a [0,90] deja fuera
    lo de abajo aunque esté sobre la misma cáscara esférica."""
    from app.modules.primitivas import Esfera, mascara_esfera
    rng = np.random.default_rng(1)
    esf = Esfera(centro=np.zeros(3), radio=2.0)
    # puntos sobre la esfera, mitad arriba y mitad abajo del ecuador
    phi = np.concatenate([rng.uniform(10, 80, 500), rng.uniform(100, 170, 500)])
    th = rng.uniform(0, 2 * np.pi, 1000)
    p = np.radians(phi)
    pts = 2.0 * np.column_stack([np.sin(p) * np.cos(th), np.sin(p) * np.sin(th),
                                 np.cos(p)])
    m = mascara_esfera(pts, esf, tolerancia=0.05, phi_min=0.0, phi_max=90.0)
    assert m[:500].all()          # los de arriba entran
    assert not m[500:].any()      # los de abajo quedan fuera


def test_tolerancia_controla_el_ancho_de_captura():
    from app.modules.primitivas import Esfera, mascara_esfera
    esf = Esfera(centro=np.zeros(3), radio=2.0)
    # puntos a 2.00, 2.10 y 2.30 del centro, sobre el eje +Z
    pts = np.array([[0, 0, 2.0], [0, 0, 2.1], [0, 0, 2.3]])
    assert mascara_esfera(pts, esf, tolerancia=0.05).tolist() == [True, False, False]
    assert mascara_esfera(pts, esf, tolerancia=0.15).tolist() == [True, True, False]
    assert mascara_esfera(pts, esf, tolerancia=0.40).tolist() == [True, True, True]
