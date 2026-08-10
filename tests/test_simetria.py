"""tests/test_simetria.py — Reparación por reflexión especular."""
import numpy as np
import pytest

from app.modules.simetria import (
    PlanoSimetria, concordancia, puntos_a_rellenar, refinar, reflejar,
)


def _media_esfera(n=4000, seed=0, y_min=0.0):
    """Media cáscara esférica: solo el lado y >= y_min está escaneado."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(4 * n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    v = v[v[:, 1] >= y_min][:n]
    return v * 5.0


def test_reflejar_es_una_involucion():
    """Reflejar dos veces devuelve el punto original: si no, el plano estaría
    trasladando además de reflejar."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-5, 5, (500, 3))
    p = PlanoSimetria(normal=[1, 2, -1], punto=[0.5, 0, 1])
    assert np.allclose(reflejar(reflejar(pts, p), p), pts)


def test_los_puntos_del_plano_no_se_mueven():
    p = PlanoSimetria(normal=[0, 1, 0], punto=[0, 3, 0])
    en_el_plano = np.array([[1.0, 3.0, 2.0], [-4.0, 3.0, 0.0]])
    assert np.allclose(reflejar(en_el_plano, p), en_el_plano)


def test_la_normal_se_normaliza_y_no_puede_ser_nula():
    assert np.isclose(np.linalg.norm(PlanoSimetria(normal=[0, 7, 0]).normal), 1.0)
    with pytest.raises(ValueError):
        PlanoSimetria(normal=[0, 0, 0])


def test_una_nube_simetrica_da_concordancia_alta():
    rng = np.random.default_rng(1)
    mitad = rng.uniform(0.2, 3.0, (2000, 3))
    pts = np.vstack([mitad, mitad * np.array([1, -1, 1])])   # simétrica en y=0
    c = concordancia(pts, PlanoSimetria(normal=[0, 1, 0], punto=[0, 0, 0]))
    assert c > 0.95


def test_un_plano_desplazado_baja_la_concordancia():
    """Es la propiedad que hace útil la métrica: un plano mal puesto fabrica una
    copia desplazada, y eso se detecta como concordancia baja, no alta."""
    rng = np.random.default_rng(2)
    mitad = rng.uniform(0.2, 3.0, (2000, 3))
    pts = np.vstack([mitad, mitad * np.array([1, -1, 1])])
    bueno = concordancia(pts, PlanoSimetria(normal=[0, 1, 0], punto=[0, 0, 0]))
    malo = concordancia(pts, PlanoSimetria(normal=[0, 1, 0], punto=[0, 1.5, 0]))
    assert bueno > 0.95 and malo < 0.35


def test_refinar_recupera_un_plano_colocado_a_ojo():
    """El caso de uso real: el usuario lo pone más o menos y el algoritmo lo
    precisa."""
    rng = np.random.default_rng(3)
    mitad = rng.uniform(0.3, 3.0, (3000, 3))
    pts = np.vstack([mitad, mitad * np.array([1, -1, 1])])

    a_ojo = PlanoSimetria(normal=[0.12, 1.0, -0.08], punto=[0, 0.25, 0])
    antes = concordancia(pts, a_ojo)
    refinado, despues = refinar(pts, a_ojo, max_iter=400)

    assert despues > antes
    assert despues > 0.90
    assert abs(refinado.normal @ np.array([0.0, 1.0, 0.0])) > 0.99
    assert abs(refinado.punto @ refinado.normal) < 0.05


def test_refinar_nunca_devuelve_algo_peor():
    """Más vale respetar lo que puso el usuario que entregarle un plano peor."""
    rng = np.random.default_rng(4)
    pts = rng.uniform(-3, 3, (1500, 3))          # sin simetría: no hay qué mejorar
    p0 = PlanoSimetria(normal=[0, 1, 0], punto=[0, 0, 0])
    c0 = concordancia(pts, p0)
    _, c1 = refinar(pts, p0, max_iter=60)
    assert c1 >= c0


def test_el_relleno_son_los_reflejos_sin_contraparte():
    """Media esfera: al reflejarla, la mitad que falta no tiene contraparte y
    es exactamente lo que hay que rellenar."""
    pts = _media_esfera(n=3000)
    p = PlanoSimetria(normal=[0, 1, 0], punto=[0, 0, 0])
    nuevos = puntos_a_rellenar(pts, p, tolerancia=0.05)

    assert len(nuevos) > 0.8 * len(pts)          # casi todo el lado que falta
    assert (nuevos[:, 1] <= 0.05).mean() > 0.95  # y caen del lado vacío


def test_una_nube_completa_no_genera_relleno():
    """Si ya está todo escaneado, la reparación no debe inventar nada."""
    rng = np.random.default_rng(5)
    v = rng.normal(size=(4000, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    pts = v * 5.0
    nuevos = puntos_a_rellenar(pts, PlanoSimetria(normal=[0, 1, 0]), tolerancia=0.30)
    assert len(nuevos) < 0.05 * len(pts)


def test_la_zona_acota_donde_aplica_la_simetria():
    """Sin acotar, una estructura simétrica solo en parte genera material falso
    en el resto."""
    pts = _media_esfera(n=3000)
    p = PlanoSimetria(normal=[0, 1, 0], punto=[0, 0, 0])
    zona = ([-6.0, -6.0, 0.0], [6.0, 6.0, 6.0])      # solo la parte alta

    todo = puntos_a_rellenar(pts, p, zona=None)
    acotado = puntos_a_rellenar(pts, p, zona=zona)
    assert 0 < len(acotado) < len(todo)
    assert (acotado[:, 2] >= -0.01).all()            # el reflejo conserva la z


def test_la_zona_vacia_no_genera_nada():
    pts = _media_esfera(n=1000)
    zona = ([100.0, 100.0, 100.0], [101.0, 101.0, 101.0])
    assert len(puntos_a_rellenar(pts, PlanoSimetria(), zona=zona)) == 0
