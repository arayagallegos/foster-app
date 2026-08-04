"""
primitivas.py — Motor de segmentación interactiva por primitivas.

Funciones puras (solo numpy): reciben la nube y los parámetros que el usuario
manipuló en el visor, y devuelven la primitiva ajustada o la máscara de puntos
capturados. No sabe nada de Qt ni de PyVista, así que se puede probar sin abrir
la ventana.

Patrón de uso (el mismo para todas las primitivas):

    1. El usuario coloca la figura MÁS O MENOS donde va  ->  centro/radio aprox.
    2. `ajustar_*` refina esos parámetros con RANSAC sobre los puntos cercanos.
    3. `mascara_*` decide qué puntos quedan capturados, según:
         - `tolerancia`: cuán cerca de la superficie debe estar un punto
         - recorte de la primitiva (para la esfera, el rango angular phi)

Por qué hacen falta los dos controles (medido sobre el Observatorio):
  - El faldón de la cúpula se desvía hasta 36 cm de la esfera: con tolerancia
    fija de 8 cm se pierde un tercio de esa zona.
  - Las compuertas están a +43 cm de la esfera, pero en phi 14-87 grados,
    mientras el faldón está en phi > 90. Sin recorte angular NO existe una
    tolerancia que capture el faldón sin absorber las compuertas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.modules.segmentation import fit_sphere_ransac


@dataclass(frozen=True)
class Esfera:
    centro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radio: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "centro", np.asarray(self.centro, dtype=float))


def _phi(pts: np.ndarray, esfera: Esfera) -> tuple[np.ndarray, np.ndarray]:
    """Distancia al centro y ángulo polar (grados, 0 = cúspide) de cada punto."""
    v = pts - esfera.centro
    d = np.linalg.norm(v, axis=1)
    seguro = np.maximum(d, 1e-9)
    return d, np.degrees(np.arccos(np.clip(v[:, 2] / seguro, -1.0, 1.0)))


def ajustar_esfera(
    pts: np.ndarray,
    centro_aprox,
    radio_aprox: float,
    margen: float | None = None,
    n_iters: int = 800,
    seed: int = 0,
    max_puntos: int = 25_000,
) -> Esfera:
    """Refina la esfera colocada a ojo por el usuario.

    Solo se ajusta con los puntos de una BANDA alrededor de la esfera aproximada
    (`margen`); si se usara toda la nube, RANSAC podría irse a otra superficie.
    Por defecto la banda es el 15 % del radio, con un mínimo de 40 cm.

    `max_puntos` submuestrea la banda antes de ajustar. Es lo que hace la
    herramienta usable: el costo de RANSAC crece con el número de puntos, pero
    el resultado no mejora. Medido sobre la nube del Observatorio (852 k puntos):

        puntos    tiempo     radio
        852.147   24.797 ms  4.656
         25.000      373 ms  4.644

    66 veces más rápido, con 1.2 cm de diferencia en el radio. Se ajusta con la
    submuestra y la máscara se aplica después a TODOS los puntos, así que no se
    pierde ni un punto en la captura.
    """
    centro_aprox = np.asarray(centro_aprox, dtype=float)
    if margen is None:
        margen = max(0.4, radio_aprox * 0.15)

    d = np.linalg.norm(pts - centro_aprox, axis=1)
    cerca = np.abs(d - radio_aprox) < margen
    if cerca.sum() < 100:
        raise RuntimeError(
            f"Solo {int(cerca.sum())} puntos cerca de la esfera: acércala a la "
            "superficie o agranda el margen."
        )

    banda = pts[cerca]
    if len(banda) > max_puntos:
        idx = np.random.default_rng(seed).choice(len(banda), max_puntos,
                                                 replace=False)
        banda = banda[idx]

    centro, radio, _ = fit_sphere_ransac(
        banda, eps=margen * 0.25, n_iters=n_iters, seed=seed
    )
    return Esfera(centro=np.asarray(centro), radio=float(radio))


def mascara_esfera(
    pts: np.ndarray,
    esfera: Esfera,
    tolerancia: float = 0.08,
    phi_min: float = 0.0,
    phi_max: float = 180.0,
) -> np.ndarray:
    """Puntos capturados por la esfera: los que están a menos de `tolerancia`
    de su cáscara Y dentro del rango angular [phi_min, phi_max].

    Separar ambos criterios es lo que permite capturar zonas que se desvían de
    la esfera (subiendo la tolerancia) sin absorber elementos vecinos que están
    en otro rango angular.
    """
    d, phi = _phi(pts, esfera)
    return ((np.abs(d - esfera.radio) < tolerancia)
            & (phi >= phi_min) & (phi <= phi_max))


def ajustar_y_capturar(
    pts: np.ndarray,
    centro_aprox,
    radio_aprox: float,
    tolerancia: float = 0.08,
    phi_min: float = 0.0,
    phi_max: float = 180.0,
) -> tuple[Esfera, np.ndarray]:
    """Conveniencia: ajusta y devuelve (esfera, máscara) de una vez.

    Ojo en la UI: al mover un slider conviene llamar solo a `mascara_esfera`
    con la esfera ya ajustada. Reajustar en cada cambio de slider es lento y
    además hace que la primitiva "salte" mientras el usuario afina.
    """
    esfera = ajustar_esfera(pts, centro_aprox, radio_aprox)
    return esfera, mascara_esfera(pts, esfera, tolerancia, phi_min, phi_max)
