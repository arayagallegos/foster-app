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

from app.modules.segmentation import (
    _fit_plane_ransac, fit_circle_ransac, fit_sphere_ransac,
)


EPS_AJUSTE = 0.05
"""Tolerancia de consenso de RANSAC, en metros: cuán lejos del modelo puede
estar un punto y aun contar como parte de la superficie.

NO debe derivarse del margen de búsqueda. El margen dice cuán mal colocó el
usuario la primitiva; `eps` dice cuán gruesa es la superficie medida. Medido
sobre un plano sintético con 1 cm de ruido y un 25 % de puntos ajenos a un solo
lado (un objeto apoyado encima, el caso real):

    eps        altura      inclinación
    0.020 m    2.000 m     0.00°        <- exacto
    0.100 m    2.051 m     1.00°
    mín. cuad. 2.064 m     0.91°

Con eps grande RANSAC admite los puntos ajenos en su consenso y pierde la
ventaja que justifica usarlo en vez de mínimos cuadrados.
"""


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
    eps: float | None = None,
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
        banda, eps=EPS_AJUSTE if eps is None else float(eps),
        n_iters=n_iters, seed=seed
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


# --------------------------------------------------------------------------- #
# Plano                                                                        #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Plano:
    """Plano `normal·x + d = 0`, acotado por un centro y un radio.

    Un plano matemático es infinito: sin acotarlo, la máscara capturaría todo lo
    que pase cerca de él en cualquier parte de la nube (el suelo del cerro, un
    muro del otro extremo). `centro` y `radio` son la parte "moldeable", el
    equivalente al recorte angular de la esfera.
    """
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    d: float = 0.0
    centro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radio: float = np.inf

    def __post_init__(self):
        n = np.asarray(self.normal, dtype=float)
        norma = float(np.linalg.norm(n))
        if norma < 1e-9:
            raise ValueError("La normal del plano no puede ser nula.")
        object.__setattr__(self, "normal", n / norma)
        object.__setattr__(self, "d", float(self.d) / norma)
        object.__setattr__(self, "centro", np.asarray(self.centro, dtype=float))

    def distancia(self, pts: np.ndarray) -> np.ndarray:
        """Distancia CON SIGNO de cada punto al plano."""
        return np.asarray(pts) @ self.normal + self.d


def ajustar_plano(
    pts: np.ndarray,
    punto_aprox,
    normal_aprox,
    radio: float,
    margen: float | None = None,
    n_iters: int = 800,
    seed: int = 0,
    max_puntos: int = 25_000,
    eps: float | None = None,
) -> Plano:
    """Refina el plano colocado a ojo, usando solo los puntos de la losa cercana
    y dentro del radio. Igual que en la esfera se submuestrea antes de ajustar:
    RANSAC no mejora con más puntos, solo tarda más."""
    punto_aprox = np.asarray(punto_aprox, dtype=float)
    n0 = np.asarray(normal_aprox, dtype=float)
    n0 = n0 / np.linalg.norm(n0)
    if margen is None:
        margen = 0.4

    dist = (pts - punto_aprox) @ n0
    en_radio = np.linalg.norm(pts - punto_aprox, axis=1) <= radio
    cerca = (np.abs(dist) < margen) & en_radio
    if cerca.sum() < 100:
        raise RuntimeError(
            f"Solo {int(cerca.sum())} puntos cerca del plano: acércalo a la "
            "superficie, agranda el radio o el margen."
        )

    losa = pts[cerca]
    if len(losa) > max_puntos:
        idx = np.random.default_rng(seed).choice(len(losa), max_puntos, replace=False)
        losa = losa[idx]

    normal, d, _ = _fit_plane_ransac(losa,
                                     eps=EPS_AJUSTE if eps is None else float(eps),
                                     n_iters=n_iters, seed=seed)
    # RANSAC devuelve la normal con signo arbitrario; se alinea con la que puso
    # el usuario, para que "el lado positivo" siga siendo el que él ve arriba.
    if normal @ n0 < 0:
        normal, d = -normal, -d
    return Plano(normal=normal, d=d, centro=punto_aprox, radio=float(radio))


def mascara_plano(pts: np.ndarray, plano: Plano, tolerancia: float = 0.08,
                  lado: str = "ambos") -> np.ndarray:
    """Puntos a menos de `tolerancia` del plano y dentro del disco de radio
    `plano.radio` centrado en `plano.centro`.

    `lado` permite quedarse con una sola cara ("positivo" / "negativo" según la
    normal), para elementos planos delgados cuyas dos caras caen juntas con
    tolerancia simétrica (un muro, un panel).

    Ojo con el alcance: cuando la nube viene de varios scans y se sabe desde
    dónde se tomó cada uno, separar las caras por PROCEDENCIA DE SCAN es mejor
    criterio, porque usa información que la geometría no tiene (dónde estaba el
    equipo). `lado` es el recurso cuando esa información no existe.
    """
    pts = np.asarray(pts)
    dist = plano.distancia(pts)
    if lado == "positivo":
        cerca = (dist >= 0) & (dist < tolerancia)
    elif lado == "negativo":
        cerca = (dist <= 0) & (dist > -tolerancia)
    elif lado == "ambos":
        cerca = np.abs(dist) < tolerancia
    else:
        raise ValueError(f"Lado desconocido: {lado!r}")
    if np.isfinite(plano.radio):
        # la distancia se mide DENTRO del plano, no en el espacio: si no, el
        # disco se deformaría en un elipsoide al inclinar el plano
        v = pts - plano.centro
        en_plano = v - np.outer(v @ plano.normal, plano.normal)
        cerca = cerca & (np.linalg.norm(en_plano, axis=1) <= plano.radio)
    return cerca


# --------------------------------------------------------------------------- #
# Cilindro                                                                     #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Cilindro:
    """Cilindro infinito de eje (`punto`, `eje`) y radio dado.

    Como en la esfera, la cáscara se acota: `h_min`/`h_max` a lo largo del eje y
    `theta_min`/`theta_max` alrededor. Sin eso, capturar el tambor se llevaría
    cualquier cosa alineada con él más arriba o más abajo.
    """
    punto: np.ndarray = field(default_factory=lambda: np.zeros(3))
    eje: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    radio: float = 1.0

    def __post_init__(self):
        e = np.asarray(self.eje, dtype=float)
        norma = float(np.linalg.norm(e))
        if norma < 1e-9:
            raise ValueError("El eje del cilindro no puede ser nulo.")
        object.__setattr__(self, "eje", e / norma)
        object.__setattr__(self, "punto", np.asarray(self.punto, dtype=float))

    def base(self) -> tuple[np.ndarray, np.ndarray]:
        """Dos vectores unitarios perpendiculares al eje y entre sí.

        `u` marca el ángulo 0. Se toma el eje X del mundo proyectado sobre el
        plano del cilindro, de modo que para un cilindro vertical el ángulo 0
        apunta al +X y el 90 al +Y. Cualquier base ortonormal serviría para las
        matemáticas, pero el usuario mueve un slider de ángulo y necesita que el
        origen sea predecible. Si el eje es casi paralelo a X se usa Y, que en
        ese caso no degenera.
        """
        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(self.eje @ ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        u = ref - (ref @ self.eje) * self.eje
        u = u / np.linalg.norm(u)
        return u, np.cross(self.eje, u)

    def coords_locales(self, pts: np.ndarray):
        """(radio, altura sobre el eje, ángulo en grados) de cada punto."""
        u, v = self.base()
        rel = np.asarray(pts) - self.punto
        h = rel @ self.eje
        x, y = rel @ u, rel @ v
        return np.hypot(x, y), h, np.degrees(np.arctan2(y, x)) % 360.0


def ajustar_cilindro(
    pts: np.ndarray,
    punto_aprox,
    eje,
    radio_aprox: float,
    h_min: float = -np.inf,
    h_max: float = np.inf,
    margen: float | None = None,
    n_iters: int = 800,
    seed: int = 0,
    max_puntos: int = 25_000,
    eps: float | None = None,
) -> Cilindro:
    """Refina el cilindro proyectando la banda sobre el plano perpendicular al
    eje y ajustando un círculo (RANSAC + Kåsa).

    La DIRECCIÓN del eje no se refina: se respeta la que puso el usuario. Con
    una sola sección la dirección queda mal condicionada, y en una estructura
    construida el usuario sabe mejor que el algoritmo si el elemento es
    vertical. Si hiciera falta, se refina después con PCA de los inliers.
    """
    cil0 = Cilindro(punto=punto_aprox, eje=eje, radio=float(radio_aprox))
    if margen is None:
        margen = max(0.4, radio_aprox * 0.15)

    r, h, _ = cil0.coords_locales(pts)
    cerca = (np.abs(r - radio_aprox) < margen) & (h >= h_min) & (h <= h_max)
    if cerca.sum() < 100:
        raise RuntimeError(
            f"Solo {int(cerca.sum())} puntos cerca del cilindro: acércalo a la "
            "superficie o agranda el margen."
        )

    banda = pts[cerca]
    if len(banda) > max_puntos:
        idx = np.random.default_rng(seed).choice(len(banda), max_puntos, replace=False)
        banda = banda[idx]

    u, v = cil0.base()
    rel = banda - cil0.punto
    xy = np.column_stack([rel @ u, rel @ v])
    cx, cy, radio, _ = fit_circle_ransac(
        xy, eps=EPS_AJUSTE if eps is None else float(eps),
        n_iters=n_iters, seed=seed)
    return Cilindro(punto=cil0.punto + cx * u + cy * v,
                    eje=cil0.eje, radio=float(radio))


def mascara_cilindro(
    pts: np.ndarray,
    cilindro: Cilindro,
    tolerancia: float = 0.08,
    h_min: float = -np.inf,
    h_max: float = np.inf,
    theta_min: float = 0.0,
    theta_max: float = 360.0,
) -> np.ndarray:
    """Puntos a menos de `tolerancia` de la cáscara, dentro del tramo de eje
    [h_min, h_max] y del sector angular [theta_min, theta_max].

    El sector admite cruzar el 0 (p. ej. de 350 a 10 grados): si theta_min es
    mayor que theta_max se toma la unión, que es lo que el usuario espera al
    arrastrar un sector por encima del origen.
    """
    r, h, th = cilindro.coords_locales(pts)
    m = (np.abs(r - cilindro.radio) < tolerancia) & (h >= h_min) & (h <= h_max)
    if theta_min <= theta_max:
        return m & (th >= theta_min) & (th <= theta_max)
    return m & ((th >= theta_min) | (th <= theta_max))


# --------------------------------------------------------------------------- #
# Cono                                                                         #
# --------------------------------------------------------------------------- #

def _fit_recta_ransac(xy: np.ndarray, eps: float, n_iters: int = 800,
                      seed: int = 0) -> tuple[float, float]:
    """RANSAC de recta y = a + b·x. Devuelve (a, b).

    Dos puntos definen el candidato; gana el de mayor consenso y se refina por
    mínimos cuadrados sobre sus inliers — igual que Kåsa refina el círculo.
    """
    rng = np.random.default_rng(seed)
    n = len(xy)
    if n < 2:
        raise RuntimeError("Muy pocos puntos para ajustar una recta.")
    mejor_n, mejor = 0, None
    for _ in range(n_iters):
        i, j = rng.choice(n, 2, replace=False)
        (x1, y1), (x2, y2) = xy[i], xy[j]
        if abs(x2 - x1) < 1e-9:
            continue
        b = (y2 - y1) / (x2 - x1)
        a = y1 - b * x1
        cuenta = int((np.abs(xy[:, 1] - (a + b * xy[:, 0])) < eps).sum())
        if cuenta > mejor_n:
            mejor_n, mejor = cuenta, (a, b)
    if mejor is None:
        raise RuntimeError("RANSAC no encontró una recta con soporte suficiente.")
    a, b = mejor
    dentro = np.abs(xy[:, 1] - (a + b * xy[:, 0])) < eps
    b, a = np.polyfit(xy[dentro, 0], xy[dentro, 1], 1)
    return float(a), float(b)


@dataclass(frozen=True)
class Cono:
    """Cono de eje (`punto`, `eje`), con radio `r0` en h=0 que crece a razón
    `pendiente` por metro de altura.

    Se parametriza así, y no por vértice y ángulo, porque es la forma en que se
    ajusta: con el eje dado, el radio de un cono crece LINEALMENTE con la
    altura, de modo que ajustar el cono es ajustar una recta. Un cilindro es el
    caso `pendiente = 0`.
    """
    punto: np.ndarray = field(default_factory=lambda: np.zeros(3))
    eje: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    r0: float = 1.0
    pendiente: float = 0.0

    def __post_init__(self):
        e = np.asarray(self.eje, dtype=float)
        norma = float(np.linalg.norm(e))
        if norma < 1e-9:
            raise ValueError("El eje del cono no puede ser nulo.")
        object.__setattr__(self, "eje", e / norma)
        object.__setattr__(self, "punto", np.asarray(self.punto, dtype=float))

    def base(self):
        return Cilindro(punto=self.punto, eje=self.eje).base()

    def coords_locales(self, pts: np.ndarray):
        return Cilindro(punto=self.punto, eje=self.eje).coords_locales(pts)

    def radio_en(self, h):
        return self.r0 + self.pendiente * np.asarray(h)

    @property
    def semiangulo(self) -> float:
        """Semiángulo de apertura, en grados."""
        return float(np.degrees(np.arctan(abs(self.pendiente))))

    def vertice(self) -> np.ndarray | None:
        """Punto donde el radio se anula. None si es un cilindro."""
        if abs(self.pendiente) < 1e-9:
            return None
        return self.punto + self.eje * (-self.r0 / self.pendiente)


def ajustar_cono(
    pts: np.ndarray,
    punto_aprox,
    eje,
    radio_aprox: float,
    h_min: float = -np.inf,
    h_max: float = np.inf,
    margen: float | None = None,
    n_iters: int = 800,
    seed: int = 0,
    max_puntos: int = 25_000,
    eps: float | None = None,
) -> Cono:
    """Refina el cono ajustando radio contra altura con RANSAC de recta.

    Igual que en el cilindro, la DIRECCIÓN del eje la pone el usuario y no se
    refina: es el dato que el algoritmo no puede recuperar bien de una nube
    parcial, y el usuario sí conoce.
    """
    cil0 = Cilindro(punto=punto_aprox, eje=eje, radio=float(radio_aprox))
    if margen is None:
        margen = max(0.4, radio_aprox * 0.30)   # más ancho que el cilindro: el
                                                # radio varía a lo largo del eje
    r, h, _ = cil0.coords_locales(pts)
    cerca = (np.abs(r - radio_aprox) < margen) & (h >= h_min) & (h <= h_max)
    if cerca.sum() < 100:
        raise RuntimeError(
            f"Solo {int(cerca.sum())} puntos cerca del cono: acércalo a la "
            "superficie o agranda el margen."
        )
    idx = np.where(cerca)[0]
    if len(idx) > max_puntos:
        idx = np.random.default_rng(seed).choice(idx, max_puntos, replace=False)

    a, b = _fit_recta_ransac(
        np.column_stack([h[idx], r[idx]]),
        eps=EPS_AJUSTE if eps is None else float(eps),
        n_iters=n_iters, seed=seed,
    )
    return Cono(punto=cil0.punto, eje=cil0.eje, r0=a, pendiente=b)


def con_apertura(cono: Cono, semiangulo_grados: float,
                 h_min: float, h_max: float) -> Cono:
    """Devuelve el mismo cono con otra apertura, GIRANDO EN TORNO A LA MITAD
    del tramo [h_min, h_max].

    Pivotar en la mitad es lo que hace usable el control: si se cambiara solo la
    pendiente dejando `r0` fijo, al abrir el cono la superficie se despegaría de
    los datos y habría que recolocar todo. Girando por el centro, el cono sigue
    apoyado donde el usuario lo puso.
    """
    if not np.isfinite([h_min, h_max]).all():
        raise ValueError("El tramo de eje debe ser finito para girar el cono.")
    h_medio = 0.5 * (h_min + h_max)
    r_medio = float(cono.radio_en(h_medio))
    pend = float(np.tan(np.radians(semiangulo_grados)))
    return Cono(punto=cono.punto, eje=cono.eje,
                r0=r_medio - pend * h_medio, pendiente=pend)


def mascara_cono(
    pts: np.ndarray,
    cono: Cono,
    tolerancia: float = 0.08,
    h_min: float = -np.inf,
    h_max: float = np.inf,
    theta_min: float = 0.0,
    theta_max: float = 360.0,
) -> np.ndarray:
    """Puntos a menos de `tolerancia` de la superficie del cono, dentro del
    tramo de eje y del sector angular.

    La distancia se mide PERPENDICULAR a la superficie, no en horizontal: en un
    cono inclinado la diferencia de radios sobreestima la distancia real por un
    factor 1/cos(semiángulo), y la tolerancia dejaría de significar metros.
    """
    r, h, th = cono.coords_locales(pts)
    dist = np.abs(r - cono.radio_en(h)) / np.sqrt(1.0 + cono.pendiente ** 2)
    m = (dist < tolerancia) & (h >= h_min) & (h <= h_max)
    if theta_min <= theta_max:
        return m & (th >= theta_min) & (th <= theta_max)
    return m & ((th >= theta_min) | (th <= theta_max))
