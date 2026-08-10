"""
simetria.py — Reparación de nubes de puntos por reflexión especular.

Completa zonas mal escaneadas reflejando material real del lado bien cubierto.
La entrada es una nube y la salida también: esto es *reparación*, no
reconstrucción — no genera superficie.

## La métrica: concordancia, no vacío

La medida de que el plano es correcto es qué fracción de los puntos reflejados
cae **encima** de puntos reales. Es lo contrario de lo intuitivo: un porcentaje
alto de reflejados en zona vacía significa que el plano está **mal** puesto,
porque está fabricando una copia desplazada de la estructura.

Medido sobre el Observatorio: compuertas 91–95 %, cúpula 88 %.

## Por qué el plano lo coloca el usuario

La detección automática del plano de simetría **falla** en datos reales: el
histograma de distancias no es bimodal. Lo que funciona es colocación gruesa a
mano más refinamiento automático — el mismo patrón que las primitivas: el
usuario aporta la intuición, el algoritmo la precisión.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TOLERANCIA = 0.05
"""Distancia (m) a la que un punto reflejado se considera "encima" de uno real.
5 cm es el orden del ruido de medición más el error de colocación del plano."""

MAX_PUNTOS_REFINE = 12_000
"""Submuestra para el refinamiento. Nelder-Mead evalúa la concordancia cientos
de veces; sobre la nube completa cada evaluación sería una consulta KD-tree de
millones de puntos. El plano resultante se aplica después a todos."""


@dataclass(frozen=True)
class PlanoSimetria:
    """Plano de reflexión, dado por una normal unitaria y un punto."""
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    punto: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        n = np.asarray(self.normal, dtype=float)
        norma = float(np.linalg.norm(n))
        if norma < 1e-9:
            raise ValueError("La normal del plano de simetría no puede ser nula.")
        object.__setattr__(self, "normal", n / norma)
        object.__setattr__(self, "punto", np.asarray(self.punto, dtype=float))


def reflejar(pts: np.ndarray, plano: PlanoSimetria) -> np.ndarray:
    """Refleja los puntos respecto del plano."""
    pts = np.asarray(pts, dtype=float)
    return pts - 2.0 * np.outer((pts - plano.punto) @ plano.normal, plano.normal)


def _a_params(plano: PlanoSimetria) -> np.ndarray:
    """(normal, punto) -> (theta, phi, d). Tres números en vez de seis: el plano
    tiene tres grados de libertad, y optimizar sobre seis dejaría direcciones
    sin efecto que confunden al optimizador."""
    n = plano.normal
    return np.array([np.arctan2(n[1], n[0]),
                     np.arccos(np.clip(n[2], -1.0, 1.0)),
                     float(plano.punto @ n)])


def _de_params(q) -> PlanoSimetria:
    th, ph, d = q
    n = np.array([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)])
    return PlanoSimetria(normal=n, punto=n * d)


def _submuestra(pts: np.ndarray, maximo: int, seed: int) -> np.ndarray:
    if len(pts) <= maximo:
        return pts
    idx = np.random.default_rng(seed).choice(len(pts), maximo, replace=False)
    return pts[idx]


def _fuente(pts: np.ndarray, zona) -> np.ndarray:
    """Puntos que se van a reflejar. `zona` es una caja (min, max) que acota
    dónde aplica la simetría: sin ella, una estructura simétrica solo en parte
    generaría material falso en el resto."""
    if zona is None:
        return pts
    mn, mx = np.asarray(zona[0], float), np.asarray(zona[1], float)
    return pts[np.all((pts >= mn) & (pts <= mx), axis=1)]


def concordancia(
    pts: np.ndarray,
    plano: PlanoSimetria,
    tolerancia: float = TOLERANCIA,
    zona=None,
    max_puntos: int = MAX_PUNTOS_REFINE,
    seed: int = 0,
) -> float:
    """Fracción de los puntos reflejados que cae sobre puntos reales.

    Se mide contra TODA la nube, aunque solo se refleje la zona: el destino de
    un reflejo puede caer fuera de la caja de origen.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(pts, dtype=float)
    fuente = _fuente(pts, zona)
    if len(fuente) == 0 or len(pts) == 0:
        return 0.0
    muestra = _submuestra(fuente, max_puntos, seed)
    d, _ = cKDTree(pts).query(reflejar(muestra, plano), workers=-1)
    return float((d < tolerancia).mean())


def refinar(
    pts: np.ndarray,
    plano: PlanoSimetria,
    tolerancia: float = TOLERANCIA,
    zona=None,
    max_puntos: int = MAX_PUNTOS_REFINE,
    seed: int = 0,
    max_iter: int = 300,
) -> tuple[PlanoSimetria, float]:
    """Ajusta el plano maximizando la concordancia, partiendo del que puso el
    usuario. Devuelve (plano refinado, concordancia).

    NO se optimiza la concordancia directamente. La concordancia es una
    *fracción de puntos dentro de un umbral*: cambia a saltos y su paisaje es
    plano a trozos, así que el optimizador no ve hacia dónde moverse y se queda
    donde empezó (medido: se estancaba en 0.07 partiendo de un plano puesto a
    ojo). Se optimiza en su lugar la **distancia media truncada** de cada
    reflejo a su punto real más cercano, que sí varía de forma continua; el
    truncamiento evita que los reflejos sin contraparte —que son justamente los
    huecos a rellenar— dominen la suma.

    La concordancia se sigue usando para *reportar* y para decidir si el
    refinamiento sirvió: es la métrica interpretable.

    Si el refinamiento empeora el resultado se devuelve el plano original: más
    vale respetar lo que puso el usuario que entregarle algo peor.
    """
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree

    pts = np.asarray(pts, dtype=float)
    inicial = concordancia(pts, plano, tolerancia, zona, max_puntos, seed)

    fuente = _fuente(pts, zona)
    if len(fuente) == 0 or len(pts) == 0:
        return plano, inicial
    muestra = _submuestra(fuente, max_puntos, seed)
    arbol = cKDTree(pts)
    tope = 3.0 * tolerancia

    def objetivo(q):
        # workers=-1: el refinamiento hace cientos de consultas y es lo único
        # que hace lenta la operación; paralelizarlas la vuelve usable
        d, _ = arbol.query(reflejar(muestra, _de_params(q)), workers=-1)
        return float(np.minimum(d, tope).mean())

    q0 = _a_params(plano)
    # simplex inicial amplio: con uno pequeño el optimizador explora solo
    # perturbaciones que no cambian nada y termina de inmediato
    paso = np.array([0.20, 0.20, max(0.10, 0.05 * abs(q0[2]) + 0.10)])
    simplex = np.vstack([q0] + [q0 + np.eye(3)[i] * paso[i] for i in range(3)])

    res = minimize(objetivo, q0, method="Nelder-Mead",
                   options={"maxiter": max_iter, "xatol": 1e-4, "fatol": 1e-6,
                            "initial_simplex": simplex})
    refinado = _de_params(res.x)
    final = concordancia(pts, refinado, tolerancia, zona, max_puntos, seed)
    if final < inicial:
        return plano, inicial
    return refinado, final


def puntos_a_rellenar(
    pts: np.ndarray,
    plano: PlanoSimetria,
    tolerancia: float = TOLERANCIA,
    zona=None,
) -> np.ndarray:
    """Reflejos que NO tienen contraparte real: son los huecos que se rellenan.

    Se aplica a todos los puntos, sin submuestrear: la submuestra sirve para
    buscar el plano, pero el relleno debe usar todo el material disponible.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(pts, dtype=float)
    fuente = _fuente(pts, zona)
    if len(fuente) == 0 or len(pts) == 0:
        return np.empty((0, 3))
    refl = reflejar(fuente, plano)
    d, _ = cKDTree(pts).query(refl, workers=-1)
    return refl[d >= tolerancia]
