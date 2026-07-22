"""
segment_DBSCAN.py — Segmenta la nube (ROI) del Observatorio y mide sus dimensiones.

Uso (venv activo, desde la raíz del proyecto):
    python scripts/segment_DBSCAN.py output/cloud/foster_roi.ply
"""
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path


pcd = o3d.io.read_point_cloud("output/seg_10/resto.ply")
labels = np.array(pcd.cluster_dbscan(eps=0.10, min_points=15, print_progress=True))
print("clusters:", labels.max() + 1, " | ruido:", (labels == -1).sum())
colors = plt.get_cmap("tab20")(labels % 20)[:, :3]
colors[labels == -1] = 0.2   # ruido en gris oscuro
pcd.colors = o3d.utility.Vector3dVector(colors)
Path("output/dbscan_resto").mkdir(parents=True, exist_ok=True)
o3d.io.write_point_cloud("output/dbscan_resto/clusters.ply", pcd)
