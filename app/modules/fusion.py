"""
fusion.py — Módulo 1: Registro multi-scan y fusión LiDAR + Fotogrametría.

Pipeline de registro:
  1. Preprocesamiento: downsample + normales + FPFH features
  2. Registro grueso: RANSAC sobre features (sin necesitar posición inicial)
  3. Registro fino: ICP Point-to-Plane
  4. Pose graph optimization: corrige errores acumulados globalmente
  5. Merge final con color transfer (LiDAR + fotogrametría)
"""

from __future__ import annotations
from typing import Callable, Optional
import copy

import numpy as np
import open3d as o3d


# ------------------------------------------------------------------ #
# Parámetros de registro (ajustables según el edificio)               #
# ------------------------------------------------------------------ #

VOXEL_REG   = 0.05   # voxel para preprocesamiento de registro (5cm)
VOXEL_MERGE = 0.03   # voxel para la nube final fusionada (3cm)


# ------------------------------------------------------------------ #
# Pipeline principal                                                   #
# ------------------------------------------------------------------ #

def register_scans(
    scans: list,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> o3d.geometry.PointCloud:
    """
    Registra una lista de scans individuales en un único sistema de coordenadas.

    Args:
        scans:       Lista de (PointCloud, scan_idx) devuelta por load_e57_scans.
        progress_cb: Callback opcional (porcentaje 0-100, mensaje).

    Returns:
        Nube de puntos fusionada y registrada.
    """
    pcds = [s[0] for s in scans]
    n = len(pcds)

    if n == 0:
        raise ValueError("No hay scans para registrar.")
    if n == 1:
        return pcds[0]

    def cb(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)
        print(f"[fusion] {pct}% — {msg}")

    cb(0, f"Preprocesando {n} scans...")

    # 1. Preprocesar todos los scans
    pcds_down, fpfhs = [], []
    for i, pcd in enumerate(pcds):
        down, fpfh = _preprocess(pcd, VOXEL_REG)
        pcds_down.append(down)
        fpfhs.append(fpfh)
        cb(int(10 * (i + 1) / n), f"Features scan {i+1}/{n}")

    # 2. Construir pose graph con registro par a par
    cb(10, "Construyendo pose graph...")
    pose_graph = _build_pose_graph(pcds_down, fpfhs, cb)

    # 3. Optimización global del pose graph
    cb(70, "Optimización global (pose graph)...")
    _optimize_pose_graph(pose_graph)

    # 4. Aplicar transformaciones y combinar
    cb(80, "Aplicando transformaciones y fusionando...")
    merged = _apply_and_merge(pcds, pose_graph, VOXEL_MERGE)

    cb(100, f"Registro completo: {len(merged.points):,} puntos")
    return merged


# ------------------------------------------------------------------ #
# Preprocesamiento                                                     #
# ------------------------------------------------------------------ #

def _preprocess(pcd: o3d.geometry.PointCloud, voxel: float):
    """
    Downsample + normales + FPFH features.
    FPFH (Fast Point Feature Histograms): descriptor local de 33 dimensiones
    que captura la distribución de normales en la vecindad de cada punto.
    Se usa para encontrar correspondencias entre scans sin posición inicial.
    """
    down = pcd.voxel_down_sample(voxel)

    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel * 2,
            max_nn=30
        )
    )
    down.orient_normals_consistent_tangent_plane(k=15)

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel * 5,
            max_nn=100
        )
    )
    return down, fpfh


# ------------------------------------------------------------------ #
# Registro par a par                                                   #
# ------------------------------------------------------------------ #

def _register_pair(
    src: o3d.geometry.PointCloud,
    dst: o3d.geometry.PointCloud,
    src_fpfh,
    dst_fpfh,
    voxel: float,
) -> tuple[np.ndarray, float]:
    """
    Registro de un par de scans en dos pasos:
      1. RANSAC sobre FPFH: alineación gruesa sin posición inicial
      2. ICP Point-to-Plane: refinamiento milimétrico

    Returns:
        (transformation 4x4, fitness 0-1)
    """
    # --- Paso 1: RANSAC global ---
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, dst, src_fpfh, dst_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 1.5),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )

    # --- Paso 2: ICP Point-to-Plane ---
    result_icp = o3d.pipelines.registration.registration_icp(
        src, dst,
        max_correspondence_distance=voxel * 0.4,
        init=result_ransac.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )

    return result_icp.transformation, result_icp.fitness


# ------------------------------------------------------------------ #
# Pose graph                                                           #
# ------------------------------------------------------------------ #

def _build_pose_graph(
    pcds_down: list,
    fpfhs: list,
    cb: Callable,
) -> o3d.pipelines.registration.PoseGraph:
    """
    Construye el grafo de poses registrando pares adyacentes y algunos
    pares no adyacentes (loop closures) para corregir deriva acumulada.

    Cada nodo = scan con su transformación estimada.
    Cada arista = restricción entre dos scans (cierta o incierta).
    """
    n = len(pcds_down)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(odometry)
    )

    total_pairs = 0
    for i in range(n):
        for j in range(i + 1, min(i + 3, n)):  # adyacentes y un salto
            total_pairs += 1

    pair_idx = 0
    for src_i in range(n):
        for dst_i in range(src_i + 1, min(src_i + 3, n)):
            pair_idx += 1
            pct = 10 + int(60 * pair_idx / max(total_pairs, 1))
            cb(pct, f"Registrando scans {src_i+1}↔{dst_i+1}...")

            T, fitness = _register_pair(
                pcds_down[src_i], pcds_down[dst_i],
                fpfhs[src_i], fpfhs[dst_i],
                VOXEL_REG,
            )
            print(f"[fusion]   par ({src_i},{dst_i}): fitness={fitness:.3f}")

            is_adjacent = (dst_i == src_i + 1)

            if is_adjacent:
                # Arista odométrica: actualiza la pose acumulada
                odometry = T @ odometry
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)
                    )
                )
                info = _compute_information_matrix(
                    pcds_down[src_i], pcds_down[dst_i], T, VOXEL_REG
                )
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        src_i, dst_i, T, info, uncertain=False
                    )
                )
            else:
                # Loop closure: arista incierta (puede ser ruidosa)
                info = _compute_information_matrix(
                    pcds_down[src_i], pcds_down[dst_i], T, VOXEL_REG
                )
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        src_i, dst_i, T, info, uncertain=True
                    )
                )

    return pose_graph


def _compute_information_matrix(
    src, dst, T, voxel
) -> np.ndarray:
    """
    Matriz de información para el pose graph.
    Indica cuánto confiar en esta restricción.
    """
    result = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        src, dst,
        max_correspondence_distance=voxel * 1.5,
        transformation=T,
    )
    return result


def _optimize_pose_graph(
    pose_graph: o3d.pipelines.registration.PoseGraph,
):
    """
    Optimización global Levenberg-Marquardt sobre el grafo de poses.
    Distribuye el error de registro a lo largo de todos los scans,
    eliminando la deriva acumulada del registro secuencial.
    """
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=VOXEL_REG * 1.5,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )


# ------------------------------------------------------------------ #
# Merge final                                                          #
# ------------------------------------------------------------------ #

def _apply_and_merge(
    pcds: list,
    pose_graph: o3d.pipelines.registration.PoseGraph,
    voxel_merge: float,
) -> o3d.geometry.PointCloud:
    """
    Aplica las transformaciones optimizadas a cada scan y los combina
    en una sola nube de puntos con voxel downsampling final.
    """
    merged = o3d.geometry.PointCloud()
    for i, pcd in enumerate(pcds):
        T = pose_graph.nodes[i].pose
        pcd_t = copy.deepcopy(pcd)
        pcd_t.transform(T)
        merged += pcd_t

    return merged.voxel_down_sample(voxel_merge)


# ------------------------------------------------------------------ #
# Transferencia de color LiDAR ← Fotogrametría                        #
# ------------------------------------------------------------------ #

def transfer_colors(
    lidar: o3d.geometry.PointCloud,
    photo: o3d.geometry.PointCloud,
) -> o3d.geometry.PointCloud:
    """
    Para cada punto del LiDAR, asigna el color del punto más cercano
    en la nube de fotogrametría (que tiene RGB real).
    Usa scipy cKDTree para máxima velocidad.
    """
    from scipy.spatial import cKDTree

    photo_pts = np.asarray(photo.points)
    photo_col = np.asarray(photo.colors)
    lidar_pts = np.asarray(lidar.points)

    tree = cKDTree(photo_pts)
    _, idx = tree.query(lidar_pts, k=1, workers=-1)

    result = copy.deepcopy(lidar)
    result.colors = o3d.utility.Vector3dVector(photo_col[idx])
    return result
