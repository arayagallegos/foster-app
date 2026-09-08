"""
clusters.py — Agrupación por densidad (DBSCAN) para segmentar lo que no es
una primitiva.

Complementa a `primitivas.py`: las primitivas capturan superficies con forma
conocida (la cúpula es una esfera, el tambor un cilindro); DBSCAN captura lo que
NO tiene forma analítica —muebles, enrejados, restos de vegetación— agrupando
puntos por cercanía entre sí.

A diferencia de `refine_interior.py`, aquí nada sabe del Observatorio: no se
usan centro ni radio de cúpula. Es la versión genérica, para que el laboratorio
pueda aplicarla a otra estructura.

Sobre el costo: NO depende del número de puntos, sino de la DENSIDAD relativa a
`eps`. Medido con `cluster_dbscan` de Open3D sobre una cáscara esférica:

    puntos     eps fijo 0.15     eps = 3x el espaciado
    150.000       0.97 s              0.21 s
    400.000       9.60 s              0.68 s
    900.000     439.57 s              2.03 s

Con un `eps` demasiado grande para la densidad, todo se funde en un cluster
gigante y cada consulta de vecindad se vuelve enorme: de ahí los 7 minutos. Con
`eps` escalado al espaciado, 900 mil puntos tardan 2 segundos.

De ahí las dos reglas del módulo: `eps` SIEMPRE se elige en relación al
espaciado (`espaciado_efectivo`), y la submuestra existe como red de seguridad
para nubes enormes, no porque el algoritmo no aguante.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RUIDO = -1

EPS_POR_ESPACIADO = 7.0
"""Cuántas veces el espaciado debe valer `eps`, y qué fracción de los vecinos
debe valer `min_points`. NO son cifras inventadas: salen de la corrida que dio
buen resultado sobre el interior del Observatorio (documentada en la memoria).

    nube        interior.ply, 253.744 puntos
    espaciado   0.0215 m
    eps         0.15 m   -> 0.15 / 0.0215 = 7.0 veces el espaciado
    vecinos     85       con ese eps
    min_points  20       -> 20 / 85 = 0.24 de los vecinos

Expresarlo como proporción y no como valor absoluto es lo que hace que se
traslade a otra nube: con 0.006 m de espaciado, el eps equivalente es 0.042 m,
no 0.15 m —que allí funde toda la estructura en un solo cluster.
"""
MIN_POINTS_POR_VECINOS = 0.24
MAX_PUNTOS_DBSCAN = 500_000
"""Tope de puntos que ve DBSCAN. Es una red de seguridad, no una necesidad:
con `eps` bien escalado 900 mil puntos tardan 2 s. Acota el daño si el usuario
elige un `eps` demasiado grande, caso en que el costo se dispara."""


def espaciado_medio(pts: np.ndarray, muestra: int = 5_000, seed: int = 0) -> float:
    """Distancia típica entre puntos vecinos.

    Es la referencia para elegir `eps`: por debajo del espaciado, DBSCAN no
    conecta nada y todo es ruido; muy por encima, funde la estructura entera en
    un solo cluster. Un valor entre 2 y 5 veces el espaciado suele funcionar.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), min(muestra, len(pts)), replace=False)
    arbol = cKDTree(pts)
    d, _ = arbol.query(pts[idx], k=2)      # k=1 es el punto consigo mismo
    return float(np.median(d[:, 1]))


def espaciado_efectivo(pts: np.ndarray, max_puntos: int = MAX_PUNTOS_DBSCAN,
                       seed: int = 0) -> float:
    """Espaciado de la nube TAL COMO LA VE DBSCAN, ya submuestreada.

    Medir sobre la nube completa da un valor demasiado pequeño y lleva a
    sugerir un `eps` con el que no se conecta nada: al submuestrear, el
    espaciado crece con la raíz del factor de reducción (los puntos están sobre
    una superficie). Con 2,09 M de puntos reducidos a 150 mil, 0.006 m pasan a
    ser 0.016 m: casi el triple.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) > max_puntos:
        idx = np.random.default_rng(seed).choice(len(pts), max_puntos,
                                                 replace=False)
        pts = pts[idx]
    return espaciado_medio(pts, seed=seed)


def vecinos_tipicos(pts: np.ndarray, eps: float, muestra: int = 300,
                    seed: int = 0) -> int:
    """Cuántos vecinos tiene un punto típico dentro del radio `eps`.

    Es el número con el que hay que comparar `min_points`: si `min_points` lo
    supera, ningún punto llega a ser núcleo y TODA la nube queda como ruido.

    No se puede deducir de una fórmula sin conocer la nube. Para puntos
    dispersos el conteo es (π/4)·(eps/s)² —medido: eps=3s da 7 vecinos, no los
    28 de π(eps/s)²— y en una nube voxelizada, que es una rejilla, sale bastante
    más alto. Por eso se mide sobre los datos en vez de estimarlo.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2 or eps <= 0:
        return 0
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), min(muestra, len(pts)), replace=False)
    arbol = cKDTree(pts)
    return int(np.median([len(v) for v in
                          arbol.query_ball_point(pts[idx], float(eps))]))


def clusterizar(
    pts: np.ndarray,
    eps: float = 0.15,
    min_points: int = 20,
    max_puntos: int = MAX_PUNTOS_DBSCAN,
    seed: int = 0,
) -> np.ndarray:
    """Etiqueta de cluster por punto (`RUIDO` = -1), para TODOS los puntos.

    Si la nube supera `max_puntos`, DBSCAN se ejecuta sobre una submuestra y las
    etiquetas se propagan al resto por vecino más cercano. OJO: `eps` se aplica
    entonces sobre la submuestra, cuyo espaciado es mayor — usa
    `espaciado_efectivo` para elegirlo, no `espaciado_medio`.
    """
    import open3d as o3d
    from scipy.spatial import cKDTree

    pts = np.asarray(pts, dtype=float)
    if len(pts) == 0:
        return np.empty(0, dtype=int)

    if len(pts) > max_puntos:
        idx = np.random.default_rng(seed).choice(len(pts), max_puntos,
                                                 replace=False)
    else:
        idx = np.arange(len(pts))

    muestra = pts[idx]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(muestra))
    etiquetas_muestra = np.asarray(
        pcd.cluster_dbscan(eps=float(eps), min_points=int(min_points))
    )
    if len(idx) == len(pts):
        return etiquetas_muestra

    # Se propaga solo desde puntos AGRUPADOS: si el ruido pudiera propagarse,
    # un punto suelto de la submuestra contaminaría a sus vecinos reales.
    agrupados = etiquetas_muestra != RUIDO
    if not agrupados.any():
        return np.full(len(pts), RUIDO, dtype=int)

    arbol = cKDTree(muestra[agrupados])
    _, vecino = arbol.query(pts, k=1)
    etiquetas = etiquetas_muestra[agrupados][vecino]
    etiquetas[idx] = etiquetas_muestra          # los muestreados mandan sobre sí
    return etiquetas.astype(int)


@dataclass(frozen=True)
class InfoCluster:
    """Descripción de un cluster, para que el usuario decida qué hacer con él."""
    id: int
    n_pts: int
    planaridad: float          # (λ2-λ3)/λ1: alto = superficie, bajo = lineal
    extension: np.ndarray      # tamaño de su caja envolvente, en metros
    centro: np.ndarray

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(self.extension))


def _planaridad(p: np.ndarray) -> float:
    """Cuán superficie es el cluster. Distingue un panel (alto) de un enrejado
    o un cable (bajo), que es la diferencia entre estructura y equipamiento."""
    if len(p) < 3:
        return 0.0
    c = p - p.mean(axis=0)
    w = np.linalg.eigvalsh(c.T @ c / len(p))     # ascendente
    l1, l2, l3 = w[2], w[1], w[0]
    return float((l2 - l3) / l1) if l1 > 0 else 0.0


def info_clusters(pts: np.ndarray, etiquetas: np.ndarray) -> list[InfoCluster]:
    """Un `InfoCluster` por cluster, ordenados de mayor a menor.

    El orden por tamaño importa: en una estructura escaneada los clusters
    grandes son la construcción y los chicos el ruido, así que la lista queda
    ordenada por relevancia sin que el usuario tenga que buscar.
    """
    pts = np.asarray(pts, dtype=float)
    etiquetas = np.asarray(etiquetas)
    infos = []
    for cid in np.unique(etiquetas):
        if cid == RUIDO:
            continue
        p = pts[etiquetas == cid]
        infos.append(InfoCluster(
            id=int(cid), n_pts=len(p), planaridad=_planaridad(p),
            extension=p.max(axis=0) - p.min(axis=0),
            centro=p.mean(axis=0),
        ))
    return sorted(infos, key=lambda i: i.n_pts, reverse=True)


def marcar_pequenos_como_ruido(etiquetas: np.ndarray, min_pts: int) -> np.ndarray:
    """Pasa a ruido los clusters con menos de `min_pts` puntos.

    Es la mitad de "limpieza" de la herramienta: `min_points` de DBSCAN decide
    qué es un núcleo denso, pero no impide que queden decenas de clusters
    pequeños y dispersos. Este umbral se aplica DESPUÉS, y se puede mover sin
    volver a correr DBSCAN, que es la operación cara.
    """
    etiquetas = np.asarray(etiquetas).copy()
    for cid in np.unique(etiquetas):
        if cid == RUIDO:
            continue
        m = etiquetas == cid
        if m.sum() < min_pts:
            etiquetas[m] = RUIDO
    return etiquetas


def mascara_de(etiquetas: np.ndarray, ids) -> np.ndarray:
    """Máscara booleana de los puntos que pertenecen a los clusters dados."""
    return np.isin(np.asarray(etiquetas), np.asarray(list(ids), dtype=int))
