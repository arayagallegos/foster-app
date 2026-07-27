"""refine_interior.py — Sub-segmenta la clase interior del Foster con DBSCAN.

Separa las compuertas de la ranura (clase propia 7), reasigna las vigas a la
cúpula (3) y descarta el telescopio/equipo (6). El borrado físico de los
descartados es un paso posterior (eliminar_descartados).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INTERIOR = 0
SUELO = 1
CUPULA = 3
DESCARTADO = 6
COMPUERTAS = 7


def dbscan_interior(
    pts_interior: np.ndarray, eps: float = 0.15, min_points: int = 20
) -> np.ndarray:
    """Etiqueta de cluster por punto (ruido = -1). Envuelve o3d cluster_dbscan."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_interior))
    return np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points))


@dataclass
class ClusterInfo:
    id: int
    n_pts: int
    frac_en_cascara: float   # fracción de puntos con |dist - r_cupula| < margen
    planaridad: float        # (λ2 - λ3)/λ1 de PCA: alto=superficie, bajo=lineal


def _planaridad(p: np.ndarray) -> float:
    if len(p) < 3:
        return 0.0
    c = p - p.mean(axis=0)
    cov = c.T @ c / len(p)
    w = np.linalg.eigvalsh(cov)          # ascendente: w[0] <= w[1] <= w[2]
    l1, l2, l3 = w[2], w[1], w[0]
    return float((l2 - l3) / l1) if l1 > 0 else 0.0


def info_clusters(
    pts_interior: np.ndarray,
    cluster_labels: np.ndarray,
    centro_cupula,
    r_cupula: float,
    margen_cascara: float = 0.30,
) -> list[ClusterInfo]:
    centro = np.asarray(centro_cupula, dtype=float)
    infos: list[ClusterInfo] = []
    for cid in sorted(c for c in np.unique(cluster_labels) if c != -1):
        p = pts_interior[cluster_labels == cid]
        d = np.linalg.norm(p - centro, axis=1)
        frac = float(np.mean(np.abs(d - r_cupula) < margen_cascara))
        infos.append(ClusterInfo(int(cid), len(p), frac, _planaridad(p)))
    return infos


@dataclass
class RefineConfig:
    eps: float = 0.15
    min_points: int = 20
    margen_cascara: float = 0.30
    min_frac_cascara: float = 0.15
    min_planaridad_compuerta: float = 0.3   # panel curvo ~0.49 vs enrejado ~0.01


def clasificar_clusters(
    infos: list[ClusterInfo], id_compuertas: int | None, config: RefineConfig
) -> dict[int, int]:
    adosados = [i for i in infos if i.frac_en_cascara >= config.min_frac_cascara]
    if id_compuertas is None:
        cands = [i for i in adosados
                 if i.planaridad >= config.min_planaridad_compuerta]
        if cands:
            id_compuertas = max(cands, key=lambda i: i.n_pts).id
    mapa: dict[int, int] = {}
    for i in infos:
        if i.frac_en_cascara < config.min_frac_cascara:
            mapa[i.id] = DESCARTADO          # no toca el domo -> telescopio/equipo
        elif i.id == id_compuertas:
            mapa[i.id] = COMPUERTAS          # objeto aparte
        else:
            mapa[i.id] = CUPULA              # vigas/costillas -> esqueleto del domo
    return mapa


def refine_interior(
    pts: np.ndarray,
    labels: np.ndarray,
    params,
    config: RefineConfig = RefineConfig(),
    id_compuertas: int | None = None,
) -> np.ndarray:
    """Reetiqueta los puntos de interior(0): compuertas(7)/cúpula(3)/descartado(6).
    El ruido de DBSCAN queda interior(0). No modifica las demás clases."""
    labels = np.asarray(labels).copy()
    idx = np.where(labels == INTERIOR)[0]
    if len(idx) == 0:
        return labels
    pts_int = pts[idx]
    cl = dbscan_interior(pts_int, config.eps, config.min_points)
    infos = info_clusters(pts_int, cl, params.centro_cupula, params.r_cupula,
                          config.margen_cascara)
    mapa = clasificar_clusters(infos, id_compuertas, config)
    for cid, destino in mapa.items():
        labels[idx[cl == cid]] = destino
    return labels


def reetiquetar_interior_por_clusters(
    pts: np.ndarray,
    labels: np.ndarray,
    config: RefineConfig = RefineConfig(),
    id_compuertas: int | None = None,
    ids_descartar: tuple[int, ...] = (),
    ids_cupula: tuple[int, ...] = (),
    ids_suelo: tuple[int, ...] = (),
) -> np.ndarray:
    """Reetiqueta el interior según clusters DBSCAN elegidos a mano por el usuario:
    id_compuertas -> compuertas(7), ids_descartar -> descartado(6), ids_cupula -> cúpula(3),
    ids_suelo -> suelo(1). Todo cluster no marcado (y el ruido) sigue interior(0). No
    modifica otras clases.

    Los ids provienen de la tabla de exploración: como DBSCAN es determinista para la
    misma nube y los mismos eps/min_points, los ids coinciden entre exploración y esta
    llamada (no cambies eps entre ambas)."""
    labels = np.asarray(labels).copy()
    idx = np.where(labels == INTERIOR)[0]
    if len(idx) == 0:
        return labels
    cl = dbscan_interior(pts[idx], config.eps, config.min_points)
    descart = set(ids_descartar)
    cup = set(ids_cupula)
    suelo = set(ids_suelo)
    for cid in np.unique(cl):
        if cid == -1:
            continue
        sel = idx[cl == cid]
        if cid in descart:
            labels[sel] = DESCARTADO
        elif cid == id_compuertas:
            labels[sel] = COMPUERTAS
        elif cid in cup:
            labels[sel] = CUPULA
        elif cid in suelo:
            labels[sel] = SUELO
    return labels


def eliminar_descartados(
    pts: np.ndarray, labels: np.ndarray, clases_a_eliminar: tuple[int, ...] = (DESCARTADO,)
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (pts, labels) sin los puntos de las clases indicadas. Etapa posterior
    y desacoplada del etiquetado: produce la nube estructural limpia para el FEM."""
    labels = np.asarray(labels)
    keep = ~np.isin(labels, np.asarray(clases_a_eliminar))
    return pts[keep], labels[keep]
