"""
meshing.py — Reconstrucción de malla a partir de una entidad segmentada.

Cierra el flujo de la herramienta: la segmentación entrega nubes de puntos por
entidad, y esto las convierte en superficie triangulada exportable.

El método está decidido con evidencia, no por intuición (`docs/resumen_experimentos.md` §2):

    metodo                                  invencion  puntos usados  manifold
    Poisson (Open3D)                            31 %       todos      si
    Ball Pivoting + reparacion                  0,8 %       67 %      tras 3 pasos
    Advancing Front (CGAL) + limite perimetro   0,65 %     100 %      por construccion

La clave del análisis fue medir en las DOS direcciones. Poisson daba 1,75 cm de
cobertura —aparentemente excelente— mientras fabricaba el 31 % de la superficie:
midiendo solo cobertura se habría elegido el peor método creyéndolo el mejor.

Funciones puras (numpy + scipy + el binding de CGAL): no saben nada de Qt ni de
PyVista, así que se pueden probar sin abrir la ventana.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PERIMETRO_POR_ESPACIADO = 14.0
"""Cuántas veces el espaciado medio debe valer el perímetro máximo de triángulo.

Es el parámetro que hace ganar a Advancing Front: sin él, el frente de avance
"puentea" los vacíos con triángulos estirados e inventa superficie donde no hubo
medición. Medido sobre las compuertas (`docs/resumen_experimentos.md` §2):

    max_perimetro   invencion
    sin limite        7,20 %
    0,50 m            1,72 %
    0,30 m            0,19 %

Como el resto de los parámetros de la herramienta, se expresa en proporción al
espaciado y no en metros: 0,30 m era ~14x el espaciado de ESA nube, y es la
proporción —no el valor— lo que se traslada a otra.
"""

PERIMETRO_RELLENO = 5.0
"""Perímetro máximo, en metros, de los agujeros que se cierran automáticamente.

Los agujeros pequeños son oclusiones y conviene cerrarlos; los grandes son zonas
sin cobertura y cerrarlos sería fabricar superficie. El umbral separa ambos casos
y deja los grandes abiertos a propósito.
"""

TOL_INVENCION = 0.05
"""Distancia, en metros, a partir de la cual una porción de malla se considera
inventada: superficie a más de esta distancia de todo punto realmente medido."""

RADIUS_RATIO_BOUND = 5.0
BETA = 0.52
"""Parámetros de forma de Advancing Front, en los valores recomendados por CGAL.
No se expusieron al usuario porque el barrido no mostró mejora al moverlos: quien
gobierna el resultado es el límite de perímetro."""


@dataclass(frozen=True)
class Malla:
    """Superficie triangulada. `caras` indexa filas de `vertices`."""

    vertices: np.ndarray
    caras: np.ndarray

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_caras(self) -> int:
        return len(self.caras)

    def areas(self) -> np.ndarray:
        """Área de cada triángulo."""
        v = self.vertices[self.caras]
        return 0.5 * np.linalg.norm(
            np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)

    @property
    def area(self) -> float:
        return float(self.areas().sum())


@dataclass(frozen=True)
class Fidelidad:
    """Cuánto se parece la malla a los puntos de los que salió.

    Las dos direcciones responden preguntas distintas y ninguna basta sola:
    `cobertura` dice si falta superficie donde sí hubo medición, e `invencion`
    dice si hay superficie donde no la hubo.
    """

    cobertura_media: float
    cobertura_p95: float
    invencion: float          # fracción de área a más de la tolerancia
    tolerancia: float
    resolucion: float         # suelo de precisión de la cobertura, en metros

    def resumen(self) -> str:
        return (f"cobertura {self.cobertura_media * 100:.2f} cm de media "
                f"({self.cobertura_p95 * 100:.2f} cm en el p95, "
                f"resolución {self.resolucion * 100:.2f} cm); "
                f"invención {self.invencion * 100:.2f} % de la superficie")


def espaciado_medio(pts: np.ndarray, muestra: int = 50_000,
                    seed: int = 0) -> float:
    """Distancia media al vecino más cercano.

    Sobre una submuestra: la media converge muy rápido y recorrer millones de
    puntos para estimarla no aporta precisión que se note en el resultado.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        raise ValueError("Hacen falta al menos dos puntos.")
    if len(pts) > muestra:
        idx = np.random.default_rng(seed).choice(len(pts), muestra, replace=False)
        sub = pts[idx]
    else:
        sub = pts
    d, _ = cKDTree(sub).query(sub, k=2, workers=-1)
    return float(d[:, 1].mean())


def perimetro_sugerido(pts: np.ndarray) -> float:
    """Límite de perímetro para esta nube, en metros."""
    return PERIMETRO_POR_ESPACIADO * espaciado_medio(pts)


def _cgal():
    """El binding vive fuera del paquete y se compila aparte."""
    import sys
    ruta = Path(__file__).resolve().parents[2] / "cgal_bridge"
    if str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))
    try:
        import cgal_bridge
    except ImportError as e:                                 # pragma: no cover
        raise RuntimeError(
            "No se encontró el binding de CGAL. Compílalo siguiendo "
            "cgal_bridge/COMPILACION.md.") from e
    return cgal_bridge


def reconstruir(pts: np.ndarray, max_perimetro: float | None = None,
                rellenar: bool = True,
                perimetro_relleno: float = PERIMETRO_RELLENO) -> Malla:
    """Nube de una entidad -> malla triangulada.

    `max_perimetro` en metros; si es None se deriva del espaciado. `rellenar`
    cierra los agujeros de perímetro menor que `perimetro_relleno` y deja los
    mayores abiertos, que es lo que evita fabricar superficie donde no hubo
    cobertura.
    """
    pts = np.ascontiguousarray(np.asarray(pts, dtype=np.float64))
    if len(pts) < 4:
        raise ValueError("Hacen falta al menos cuatro puntos para triangular.")
    if max_perimetro is None:
        max_perimetro = perimetro_sugerido(pts)
    if max_perimetro <= 0:
        raise ValueError("El perímetro máximo debe ser positivo.")

    cgal = _cgal()
    V, F = cgal.advancing_front(pts, RADIUS_RATIO_BOUND, BETA, float(max_perimetro))
    if len(F) == 0:
        raise ValueError(
            "La reconstrucción no produjo ningún triángulo. El límite de "
            "perímetro puede ser demasiado pequeño para la densidad de la nube.")
    malla = Malla(np.asarray(V, dtype=float), np.asarray(F, dtype=np.int32))
    if rellenar and perimetro_relleno > 0:
        malla = rellenar_agujeros(malla, perimetro_relleno)
    return malla


def rellenar_agujeros(malla: Malla, perimetro_maximo: float) -> Malla:
    """Cierra los contornos abiertos de perímetro menor que el umbral."""
    cgal = _cgal()
    V, F = cgal.fill_holes(
        np.ascontiguousarray(malla.vertices, dtype=np.float64),
        np.ascontiguousarray(malla.caras, dtype=np.int32),
        True, 2.0, float(perimetro_maximo), True, 1)
    return Malla(np.asarray(V, dtype=float), np.asarray(F, dtype=np.int32))


def agujeros(malla: Malla) -> list[dict]:
    """Perímetro y número de aristas de cada contorno abierto, de mayor a menor.

    Un "agujero" es un CONTORNO CERRADO de aristas de borde, no un hueco visual.
    El borde exterior de una superficie abierta cuenta como uno solo por dentado
    que se vea: medido sobre un parche con el borde ondulado en treinta entrantes
    y tres agujeros interiores, el contorno exterior aparece como un único
    contorno de 593 aristas. Sin esta aclaración, el recuento parece no
    corresponder con lo que se ve en pantalla.
    """
    cgal = _cgal()
    lista = [dict(h) for h in cgal.hole_sizes(
        np.ascontiguousarray(malla.vertices, dtype=np.float64),
        np.ascontiguousarray(malla.caras, dtype=np.int32))]
    return sorted(lista, key=lambda h: -h.get("perimetro", 0.0))


def diagnostico(malla: Malla) -> dict:
    """Agujeros, cierre, auto-intersecciones y orientación."""
    cgal = _cgal()
    return dict(cgal.mesh_stats(
        np.ascontiguousarray(malla.vertices, dtype=np.float64),
        np.ascontiguousarray(malla.caras, dtype=np.int32)))


def muestrear_superficie(malla: Malla, n: int, seed: int = 0) -> np.ndarray:
    """Puntos repartidos uniformemente POR ÁREA sobre la malla.

    Uniforme por área y no por triángulo: si no, los triángulos diminutos
    pesarían lo mismo que los grandes y la medida de invención quedaría
    dominada por el detalle en vez de por la superficie.
    """
    if malla.n_caras == 0 or n <= 0:
        return np.empty((0, 3), dtype=float)
    rng = np.random.default_rng(seed)
    areas = malla.areas()
    total = areas.sum()
    if total <= 0:
        return np.empty((0, 3), dtype=float)
    # Elección del triángulo por CDF acumulada del área.
    tri = np.searchsorted(np.cumsum(areas) / total, rng.random(n))
    tri = np.clip(tri, 0, malla.n_caras - 1)
    v = malla.vertices[malla.caras[tri]]
    # Coordenadas baricéntricas uniformes sobre el triángulo.
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    a, b, c = (1 - s)[:, None], (s * (1 - r2))[:, None], (s * r2)[:, None]
    return a * v[:, 0] + b * v[:, 1] + c * v[:, 2]


def fidelidad(malla: Malla, pts: np.ndarray, tolerancia: float = TOL_INVENCION,
              n_muestras: int | None = None, seed: int = 0) -> Fidelidad:
    """Compara la malla con los puntos de los que salió, en ambas direcciones.

    Medir solo una dirección engaña: un método que extrapola hacia el vacío da
    una cobertura excelente mientras fabrica un tercio de la superficie.

    Ambas se calculan sobre un muestreo de la superficie, y eso impone un SUELO
    a la cobertura: un punto que está exactamente sobre la malla queda igualmente
    a media separación de muestras del vecino más próximo. Ese suelo se estima y
    se devuelve como `resolucion`, de modo que una cobertura de su mismo orden se
    lee como "indistinguible de cero" y no como un error de la malla.

    Por omisión se eligen tantas muestras como haga falta para que su separación
    sea la tercera parte del espaciado de la nube —con lo que el suelo queda muy
    por debajo de cualquier magnitud con sentido físico—, con un tope para que
    una entidad grande no dispare el cálculo.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) == 0:
        raise ValueError("No hay puntos con los que comparar.")
    if n_muestras is None:
        # Separación de muestras objetivo: un tercio del espaciado de la nube.
        # En una superficie de area A repartida en n muestras, la separación
        # tipica es sqrt(A/n), de donde n = 9*A/espaciado^2.
        s = espaciado_medio(pts)
        objetivo = 9.0 * malla.area / max(s * s, 1e-12)
        n_muestras = int(min(300_000, max(20_000, objetivo)))

    # Las DOS direcciones se miden contra la superficie muestreada, no contra
    # los vértices. Un triángulo grande tiene tres vértices y kilómetros de
    # interior: medir contra ellos daría por lejana una superficie que pasa
    # justo por debajo del punto.
    muestra = muestrear_superficie(malla, n_muestras, seed)
    if len(muestra) == 0:
        raise ValueError("La malla no tiene superficie que comparar.")

    cobertura, _ = cKDTree(muestra).query(pts, workers=-1)
    d, _ = cKDTree(pts).query(muestra, workers=-1)
    invencion = float((d > tolerancia).mean())
    resolucion = 0.5 * float(np.sqrt(malla.area / max(len(muestra), 1)))

    return Fidelidad(
        cobertura_media=float(cobertura.mean()),
        cobertura_p95=float(np.percentile(cobertura, 95)),
        invencion=invencion,
        tolerancia=float(tolerancia),
        resolucion=resolucion,
    )


def exportar(malla: Malla, ruta) -> Path:
    """Escribe la malla con normales por vértice, para que se vea sombreada."""
    import open3d as o3d

    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(malla.vertices, dtype=float)),
        o3d.utility.Vector3iVector(np.asarray(malla.caras, dtype=np.int32)))
    m.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(ruta), m):
        raise OSError(f"No se pudo escribir {ruta}")
    return ruta
