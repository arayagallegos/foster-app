"""
segmentation.py — Módulo 4: Segmentación de elementos estructurales.

Identificación automática de entidades del Observatorio por RANSAC:
suelo (plano, Open3D), tambor (círculo 2D propio, eje vertical asumido),
cúpula (esfera propia). Etiquetas: 0=resto, 1=suelo, 2=tambor, 3=cupula.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ #
# Círculo 2D (tambor, eje vertical asumido)                            #
# ------------------------------------------------------------------ #

def _circle_from_3(p: np.ndarray) -> tuple[float, float, float] | None:
    """Circuncentro de 3 puntos 2D; None si son casi colineales."""
    (x1, y1), (x2, y2), (x3, y3) = p
    a = np.array([[2 * (x2 - x1), 2 * (y2 - y1)],
                  [2 * (x3 - x1), 2 * (y3 - y1)]])
    b = np.array([x2**2 - x1**2 + y2**2 - y1**2,
                  x3**2 - x1**2 + y3**2 - y1**2])
    det = np.linalg.det(a)
    if abs(det) < 1e-9:
        return None
    cx, cy = np.linalg.solve(a, b)
    r = float(np.hypot(x1 - cx, y1 - cy))
    return float(cx), float(cy), r


def _kasa_refit(xy: np.ndarray) -> tuple[float, float, float]:
    """Ajuste de círculo por mínimos cuadrados algebraicos (método de Kåsa)."""
    a = np.column_stack([2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))])
    b = (xy**2).sum(axis=1)
    (cx, cy, t), *_ = np.linalg.lstsq(a, b, rcond=None)
    return float(cx), float(cy), float(np.sqrt(t + cx**2 + cy**2))


def fit_circle_ransac(
    xy: np.ndarray,
    eps: float = 0.05,
    n_iters: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float, np.ndarray]:
    """
    RANSAC de círculo en 2D: 3 puntos aleatorios definen un candidato; gana el
    de mayor soporte. Refinamiento final por Kåsa sobre los inliers.
    """
    rng = np.random.default_rng(seed)
    best_count, best = 0, None
    for _ in range(n_iters):
        cand = _circle_from_3(xy[rng.choice(len(xy), 3, replace=False)])
        if cand is None:
            continue
        cx, cy, r = cand
        dist = np.abs(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r)
        count = int((dist < eps).sum())
        if count > best_count:
            best_count, best = count, (cx, cy, r)
    if best is None or best_count < 0.03 * len(xy):
        raise RuntimeError("RANSAC no encontró un círculo con soporte suficiente.")

    cx, cy, r = best
    mask = np.abs(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r) < eps
    cx, cy, r = _kasa_refit(xy[mask])
    mask = np.abs(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r) < eps
    return cx, cy, r, mask


# ------------------------------------------------------------------ #
# Esfera (cúpula)                                                      #
# ------------------------------------------------------------------ #

def _sphere_from_4(p: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Esfera por 4 puntos: sistema lineal x²+y²+z² = 2c·p + t."""
    a = np.column_stack([2 * p, np.ones(4)])
    b = (p**2).sum(axis=1)
    if abs(np.linalg.det(a)) < 1e-9:
        return None
    sol = np.linalg.solve(a, b)
    c = sol[:3]
    r = float(np.sqrt(sol[3] + (c**2).sum()))
    return c, r


def _sphere_refit(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Mínimos cuadrados algebraicos de esfera sobre los inliers."""
    a = np.column_stack([2 * pts, np.ones(len(pts))])
    b = (pts**2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    c = sol[:3]
    return c, float(np.sqrt(sol[3] + (c**2).sum()))


def fit_sphere_ransac(
    pts: np.ndarray,
    eps: float = 0.08,
    n_iters: int = 2000,
    seed: int = 0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """RANSAC de esfera: 4 puntos definen el candidato; gana el de mayor soporte."""
    rng = np.random.default_rng(seed)
    best_count, best = 0, None
    for _ in range(n_iters):
        cand = _sphere_from_4(pts[rng.choice(len(pts), 4, replace=False)])
        if cand is None or not (0.1 < cand[1] < 50.0):   # descarta esferas absurdas
            continue
        c, r = cand
        dist = np.abs(np.linalg.norm(pts - c, axis=1) - r)
        count = int((dist < eps).sum())
        if count > best_count:
            best_count, best = count, (c, r)
    if best is None or best_count < 0.03 * len(pts):
        raise RuntimeError("RANSAC no encontró una esfera con soporte suficiente.")

    c, r = best
    mask = np.abs(np.linalg.norm(pts - c, axis=1) - r) < eps
    c, r = _sphere_refit(pts[mask])
    mask = np.abs(np.linalg.norm(pts - c, axis=1) - r) < eps
    return c, r, mask


# ------------------------------------------------------------------ #
# Suelo (plano con filtro de verticalidad)                             #
# ------------------------------------------------------------------ #

def segment_ground(
    pts: np.ndarray,
    eps: float = 0.05,
    max_tilt_deg: float = 15.0,
    max_attempts: int = 5,
) -> tuple[float, np.ndarray]:
    """
    Plano de suelo con segment_plane de Open3D, aceptando solo planos cuya
    normal forme < max_tilt_deg con la vertical (los dominantes no horizontales,
    p. ej. paredes, se descartan y se reintenta sobre el resto).
    """
    import open3d as o3d

    cos_max = np.cos(np.radians(max_tilt_deg))
    restantes = np.arange(len(pts))
    for _ in range(max_attempts):
        sub = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[restantes]))
        (a, b, c, _d), idx = sub.segment_plane(
            distance_threshold=eps, ransac_n=3, num_iterations=1000
        )
        normal = np.array([a, b, c])
        normal /= np.linalg.norm(normal)
        if abs(normal[2]) >= cos_max:
            mask = np.zeros(len(pts), dtype=bool)
            mask[restantes[idx]] = True
            return float(np.median(pts[mask, 2])), mask
        restantes = np.delete(restantes, idx)
    raise RuntimeError("No se encontró un plano de suelo con normal vertical.")


# ------------------------------------------------------------------ #
# Orquestador                                                          #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class SegmentationConfig:
    eps_plano: float = 0.05
    eps_circulo: float = 0.05
    eps_esfera: float = 0.08
    max_tilt_deg: float = 15.0
    margen_cupula: float = 0.5
    n_iters: int = 2000
    seed: int = 0


@dataclass(frozen=True)
class FosterParams:
    """Dimensiones medidas de la nube — la interfaz con la generación de geometría."""
    z_suelo: float
    cx: float
    cy: float
    r_tambor_ext: float
    r_tambor_int: float | None
    z_top_muro: float
    centro_cupula: tuple[float, float, float]
    r_cupula: float


@dataclass(frozen=True)
class SegmentationResult:
    labels: np.ndarray   # (N,) int: 0=resto, 1=suelo, 2=tambor, 3=cupula
    params: FosterParams


def _segunda_cara_muro(
    xy: np.ndarray, no_ring: np.ndarray, cx: float, cy: float, r_ext: float,
    cfg: SegmentationConfig,
) -> tuple[float | None, np.ndarray]:
    """
    Busca la cara interior del muro (scans interiores): un segundo círculo
    concéntrico de radio menor. Devuelve (r_int, mask_extra_sobre_xy).
    """
    sin_ring = np.where(no_ring)[0]
    if len(sin_ring) < 100:
        return None, np.zeros(len(xy), dtype=bool)
    try:
        cx2, cy2, r2, mask2 = fit_circle_ransac(
            xy[sin_ring], eps=cfg.eps_circulo, n_iters=cfg.n_iters, seed=cfg.seed
        )
    except RuntimeError:
        return None, np.zeros(len(xy), dtype=bool)
    concentrico = np.hypot(cx2 - cx, cy2 - cy) < 0.3
    espesor_plausible = 0.1 < (r_ext - r2) < 1.0
    if not (concentrico and espesor_plausible):
        return None, np.zeros(len(xy), dtype=bool)
    mask = np.zeros(len(xy), dtype=bool)
    mask[sin_ring[mask2]] = True
    return r2, mask


def segment_foster(
    pts: np.ndarray,
    config: SegmentationConfig = SegmentationConfig(),
) -> SegmentationResult:
    """
    Pipeline de identificación de entidades del Observatorio:
    suelo (plano) → tambor (círculo 2D, eje vertical) → cúpula (esfera).
    La frontera muro/cúpula es la cota del centro de la esfera (sz).
    """
    labels = np.zeros(len(pts), dtype=int)

    # 1. Suelo
    z_suelo, mask_suelo = segment_ground(
        pts, eps=config.eps_plano, max_tilt_deg=config.max_tilt_deg
    )
    labels[mask_suelo] = 1
    resto_idx = np.where(~mask_suelo)[0]
    p_resto = pts[resto_idx]

    # 2. Tambor: círculo exterior (+ cara interior si hay scans interiores)
    xy = p_resto[:, :2]
    cx, cy, r_ext, ring = fit_circle_ransac(
        xy, eps=config.eps_circulo, n_iters=config.n_iters, seed=config.seed
    )
    # La cara interior solo puede estar a altura de muro: excluir la cúpula,
    # cuya proyección XY genera anillos espurios de radio menor.
    z_top_ring = float(np.percentile(p_resto[ring, 2], 98))
    en_muro = p_resto[:, 2] < z_top_ring - 0.3
    r_int, ring_int = _segunda_cara_muro(xy, ~ring & en_muro, cx, cy, r_ext, config)
    if r_int is not None and r_int > r_ext:      # el 1er círculo fue el interior
        r_ext, r_int = r_int, r_ext
    ring_total = ring | ring_int

    # 3. Cúpula: esfera sobre los puntos altos no asignados
    z_ring_p98 = float(np.percentile(p_resto[ring_total, 2], 98))
    candidatos = np.where(p_resto[:, 2] > z_ring_p98 - config.margen_cupula)[0]
    centro, r_cupula, mask_cand = fit_sphere_ransac(
        p_resto[candidatos], eps=config.eps_esfera,
        n_iters=config.n_iters, seed=config.seed,
    )
    if np.hypot(centro[0] - cx, centro[1] - cy) > 0.5:
        raise RuntimeError(
            "La esfera de la cúpula no es coaxial con el tambor "
            f"(centro esfera {centro[:2]}, eje tambor ({cx:.2f}, {cy:.2f}))."
        )
    sz = float(centro[2])

    # 4. Asignación final (frontera muro/cúpula en z = sz)
    dist_esfera = np.abs(np.linalg.norm(p_resto - centro, axis=1) - r_cupula)
    cupula = (dist_esfera < config.eps_esfera) & (p_resto[:, 2] > sz)
    tambor = ring_total & (p_resto[:, 2] <= sz) & (p_resto[:, 2] > z_suelo + 0.05)
    labels[resto_idx[cupula]] = 3
    labels[resto_idx[tambor & ~cupula]] = 2

    params = FosterParams(
        z_suelo=z_suelo, cx=cx, cy=cy,
        r_tambor_ext=r_ext, r_tambor_int=r_int,
        z_top_muro=sz,
        centro_cupula=(float(centro[0]), float(centro[1]), sz),
        r_cupula=r_cupula,
    )
    return SegmentationResult(labels=labels, params=params)


# ------------------------------------------------------------------ #
# Salidas                                                              #
# ------------------------------------------------------------------ #

_COLORES = {           # RGB 0-1 por clase, para el visualizador y figuras
    "resto":  (0.6, 0.6, 0.6),
    "suelo":  (0.55, 0.4, 0.25),
    "tambor": (0.85, 0.2, 0.2),
    "cupula": (0.2, 0.4, 0.85),
}
_NOMBRES = {0: "resto", 1: "suelo", 2: "tambor", 3: "cupula"}


def save_segments(
    pts: np.ndarray, result: SegmentationResult, out_dir: Path
) -> dict[str, Path]:
    """Escribe un .ply coloreado por clase y parametros_foster.json en out_dir."""
    import open3d as o3d

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rutas: dict[str, Path] = {}
    for valor, nombre in _NOMBRES.items():
        sel = pts[result.labels == valor]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sel))
        pcd.paint_uniform_color(_COLORES[nombre])
        ruta = out_dir / f"{nombre}.ply"
        o3d.io.write_point_cloud(str(ruta), pcd)
        rutas[nombre] = ruta

    (out_dir / "parametros_foster.json").write_text(
        json.dumps(asdict(result.params), indent=2)
    )
    return rutas
