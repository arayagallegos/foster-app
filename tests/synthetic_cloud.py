"""tests/synthetic_cloud.py — Nube sintética del 'Observatorio' con verdad conocida.

Soporta casos difíciles opcionales (desnivel de suelo, junta cúpula-tambor realista,
contrafuertes, interior) para evaluar la precisión de la segmentación con métricas.
"""
from __future__ import annotations

import numpy as np

R_TAMBOR = 2.5
H_MURO = 3.0
R_SUELO = 6.0


def _ring(r: float, z_min: float, z_max: float, n: int, rng) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    z = rng.uniform(z_min, z_max, n)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])


def _dome(r: float, cz: float, n: int, rng) -> np.ndarray:
    """Semiesfera (muestreo uniforme en área) centrada en (0, 0, cz)."""
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    cos_phi = rng.uniform(0.0, 1.0, n)
    sin_phi = np.sqrt(1.0 - cos_phi**2)
    return np.column_stack([
        r * sin_phi * np.cos(theta),
        r * sin_phi * np.sin(theta),
        cz + r * cos_phi,
    ])


def _dome_realista(r: float, cz: float, n: int, rng) -> np.ndarray:
    """
    Cúpula con un leve sobresalto (overhang) y una FALDA que baja por debajo de
    su ecuador (z < cz). Reproduce el mecanismo real observado en la nube: la
    banda de transición de la cúpula queda bajo el plano z = sz y el corte
    horizontal rígido del orquestador la misclasifica como tambor (a cualquier
    tolerancia). La verdad de referencia de estos puntos sigue siendo cúpula.
    """
    r_over = r + 0.02                              # leve overhang, dentro de la banda
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    cos_phi = rng.uniform(-0.15, 1.0, n)           # phi hasta ~99°: falda bajo el ecuador
    sin_phi = np.sqrt(1.0 - cos_phi**2)
    return np.column_stack([
        r_over * sin_phi * np.cos(theta),
        r_over * sin_phi * np.sin(theta),
        cz + r_over * cos_phi,                     # cos_phi<0 -> z < cz (bajo el ecuador)
    ])


def _disk(r_max: float, z: float, n: int, rng) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    r = r_max * np.sqrt(rng.uniform(0.0, 1.0, n))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), np.full(n, z)])


def _contrafuertes(n_buttress: int, n_pts: int, rng) -> np.ndarray:
    """Prismas verticales pegados a la cara exterior del muro, repartidos en ángulo."""
    out = []
    for k in range(n_buttress):
        ang = 2.0 * np.pi * k / max(n_buttress, 1)
        base = np.array([R_TAMBOR * np.cos(ang), R_TAMBOR * np.sin(ang), 0.0])
        du = np.array([np.cos(ang), np.sin(ang), 0.0])           # radial
        dv = np.array([-np.sin(ang), np.cos(ang), 0.0])          # tangencial
        u = rng.uniform(0.0, 0.4, n_pts)                         # 40 cm hacia afuera
        v = rng.uniform(-0.25, 0.25, n_pts)                      # 50 cm de ancho
        z = rng.uniform(0.0, H_MURO, n_pts)
        pts = base + np.outer(u, du) + np.outer(v, dv)
        pts[:, 2] = z
        out.append(pts)
    return np.vstack(out) if out else np.empty((0, 3))


def make_synthetic_foster(
    seed: int = 0,
    noise: float = 0.01,
    outlier_frac: float = 0.2,
    r_int: float | None = None,
    pendiente_suelo_deg: float = 0.0,
    junta_realista: bool = False,
    n_contrafuertes: int = 0,
    n_interior: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    dome_fn = _dome_realista if junta_realista else _dome
    partes = [
        (_disk(R_SUELO, 0.0, 8000, rng), 1),
        (_ring(R_TAMBOR, 0.0, H_MURO, 8000, rng), 2),
        (dome_fn(R_TAMBOR, H_MURO, 6000, rng), 3),
    ]
    if r_int is not None:
        partes.append((_ring(r_int, 0.0, H_MURO, 4000, rng), 2))
    if n_contrafuertes > 0:
        partes.append((_contrafuertes(n_contrafuertes, 800, rng), 0))
    if n_interior > 0:
        interior = np.column_stack([
            rng.uniform(-R_TAMBOR * 0.7, R_TAMBOR * 0.7, n_interior),
            rng.uniform(-R_TAMBOR * 0.7, R_TAMBOR * 0.7, n_interior),
            rng.uniform(0.2, H_MURO * 0.8, n_interior),
        ])
        partes.append((interior, 0))

    pts = np.vstack([p for p, _ in partes])
    labels = np.concatenate([np.full(len(p), lab) for p, lab in partes])
    pts = pts + rng.normal(0.0, noise, pts.shape)

    # Desnivel: inclinar todo el modelo alrededor del eje Y por la pendiente del suelo
    if pendiente_suelo_deg != 0.0:
        a = np.radians(pendiente_suelo_deg)
        # z += tan(a) * x  (plano de suelo deja de ser horizontal)
        pts = pts.copy()
        pts[:, 2] = pts[:, 2] + np.tan(a) * pts[:, 0]

    if outlier_frac > 0.0:
        n_out = int(outlier_frac * len(pts))
        outliers = np.column_stack([
            rng.uniform(-6.0, 6.0, n_out),
            rng.uniform(-6.0, 6.0, n_out),
            rng.uniform(0.0, 7.0, n_out),
        ])
        pts = np.vstack([pts, outliers])
        labels = np.concatenate([labels, np.zeros(n_out, dtype=int)])

    orden = rng.permutation(len(pts))
    return pts[orden], labels[orden]


def make_synthetic_foster_dificil(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Nube con todos los casos difíciles activados (escenario de evaluación)."""
    return make_synthetic_foster(
        seed=seed,
        pendiente_suelo_deg=4.0,
        junta_realista=True,
        n_contrafuertes=6,
        n_interior=1500,
        outlier_frac=0.15,
    )
