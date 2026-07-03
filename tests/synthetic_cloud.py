"""tests/synthetic_cloud.py — Nube sintética del 'Observatorio' con verdad conocida."""
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
    cos_phi = rng.uniform(0.0, 1.0, n)          # uniforme en área para casquete
    sin_phi = np.sqrt(1.0 - cos_phi**2)
    return np.column_stack([
        r * sin_phi * np.cos(theta),
        r * sin_phi * np.sin(theta),
        cz + r * cos_phi,
    ])


def _disk(r_max: float, z: float, n: int, rng) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    r = r_max * np.sqrt(rng.uniform(0.0, 1.0, n))   # sqrt → densidad uniforme
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), np.full(n, z)])


def make_synthetic_foster(
    seed: int = 0,
    noise: float = 0.01,
    outlier_frac: float = 0.2,
    r_int: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    partes = [
        (_disk(R_SUELO, 0.0, 8000, rng), 1),
        (_ring(R_TAMBOR, 0.0, H_MURO, 8000, rng), 2),
        (_dome(R_TAMBOR, H_MURO, 6000, rng), 3),
    ]
    if r_int is not None:
        partes.append((_ring(r_int, 0.0, H_MURO, 4000, rng), 2))

    pts = np.vstack([p for p, _ in partes])
    labels = np.concatenate([np.full(len(p), lab) for p, lab in partes])
    pts = pts + rng.normal(0.0, noise, pts.shape)

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
