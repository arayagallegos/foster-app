"""
processing.py — Modulo 2: Filtrado, subsampling y edicion de la nube.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d


def crop_cloud(
    pcd: o3d.geometry.PointCloud,
    min_bound: np.ndarray,
    max_bound: np.ndarray,
) -> o3d.geometry.PointCloud:
    """
    Recorta una nube de puntos a un bounding box axis-aligned.

    Args:
        pcd:       Nube de entrada (no se modifica).
        min_bound: Array [x, y, z] del extremo minimo del box (coords mundo).
        max_bound: Array [x, y, z] del extremo maximo del box (coords mundo).

    Returns:
        Nueva nube con solo los puntos dentro del box.

    Raises:
        ValueError: Si la seleccion no contiene ningun punto.
    """
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.asarray(min_bound, dtype=np.float64),
        max_bound=np.asarray(max_bound, dtype=np.float64),
    )
    result = pcd.crop(bbox)
    if len(result.points) == 0:
        raise ValueError("La seleccion no contiene puntos.")
    return result


def apply_lasso(
    screen_pts: np.ndarray,
    valid_mask: np.ndarray,
    polygon_verts: list[tuple[int, int]],
    set_op: str,
    current_mask: np.ndarray,
) -> np.ndarray:
    if len(polygon_verts) < 3:
        raise ValueError("Se necesitan al menos 3 puntos para definir un lazo.")

    valid_ops = {"union", "intersection", "difference", "symmetric_difference"}
    if set_op not in valid_ops:
        raise ValueError(f"Operacion '{set_op}' no valida. Use: {sorted(valid_ops)}")

    from matplotlib.path import Path

    path = Path(polygon_verts)
    in_polygon = path.contains_points(screen_pts) & valid_mask

    ops = {
        "union":                lambda a, b: a | b,
        "intersection":         lambda a, b: a & b,
        "difference":           lambda a, b: a & ~b,
        "symmetric_difference": lambda a, b: a ^ b,
    }
    return ops[set_op](current_mask, in_polygon)
