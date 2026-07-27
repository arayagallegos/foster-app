"""
explora_dbscan_interior.py — Exploración manual de DBSCAN sobre una nube.

Corre cluster_dbscan con los parámetros que le pases, imprime cuántos clusters,
ruido y tamaños salen, y escribe un .ply coloreado por cluster para inspeccionar
en el visor. Sirve para elegir a ojo un eps/min_points antes de fijar el diseño.

Uso (venv activo, desde la raíz del proyecto):
    python scripts/explora_dbscan_interior.py output/perfil_real/interior.ply --eps 0.10
    python scripts/explora_dbscan_interior.py output/perfil_real/interior.ply --eps 0.20 --min-points 30

Salida: output/explora_dbscan/<nombre>_eps<eps>.ply  (ruido en gris oscuro)
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def main():
    ap = argparse.ArgumentParser(description="Exploración manual de DBSCAN")
    ap.add_argument("nube")
    ap.add_argument("--eps", type=float, default=0.10, help="radio de vecindad (m)")
    ap.add_argument("--min-points", type=int, default=20, help="mínimo de puntos por cluster")
    ap.add_argument("--out", default="output/explora_dbscan")
    ap.add_argument("--top", type=int, default=15, help="cuántos clusters listar")
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(args.nube)
    pts = np.asarray(pcd.points)
    print(f"Nube: {len(pts):,} puntos | eps={args.eps} min_points={args.min_points}")

    labels = np.array(pcd.cluster_dbscan(
        eps=args.eps, min_points=args.min_points, print_progress=True))
    n_clu = labels.max() + 1
    ruido = int((labels == -1).sum())
    print(f"\nClusters: {n_clu} | ruido: {ruido:,} ({100*ruido/len(pts):.1f}%)")

    tam = sorted(((c, int((labels == c).sum())) for c in range(n_clu)),
                 key=lambda t: -t[1])
    print(f"\n  {'cluster':>7} {'n_pts':>9} {'% del total':>11}")
    for c, n in tam[:args.top]:
        print(f"  {c:>7} {n:>9,} {100*n/len(pts):>10.1f}%")

    colors = plt.get_cmap("tab20")(labels % 20)[:, :3]
    colors[labels == -1] = 0.15
    pcd.colors = o3d.utility.Vector3dVector(colors)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.nube).stem
    ruta = outdir / f"{stem}_eps{args.eps}_mp{args.min_points}.ply"
    o3d.io.write_point_cloud(str(ruta), pcd)
    print(f"\nColoreado -> {ruta}")


if __name__ == "__main__":
    main()
