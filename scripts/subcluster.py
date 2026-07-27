"""
subcluster.py — Sub-clusteriza un cluster ya aislado (p. ej. cluster_3.ply) con un
eps más chico, para separar objetos que quedaron mezclados a eps mayor.

Dos modos:
  1) Explorar (sin --descartar): corre DBSCAN, imprime la tabla de sub-clusters y
     vuelca cada uno como archivo, para identificarlos a ojo.
        python scripts/subcluster.py output/refine_real/clusters/cluster_3.ply --eps 0.10
  2) Limpiar (con --descartar): escribe el cluster SIN los sub-clusters indicados
     (los muebles), conservando el resto y el ruido.
        python scripts/subcluster.py output/refine_real/clusters/cluster_3.ply --eps 0.10 \
            --descartar 5,8,12 --out output/refine_real/cluster_3_limpio.ply
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def _ids(texto):
    if not texto:
        return set()
    return {int(x) for x in texto.replace(" ", "").split(",") if x != ""}


def main():
    ap = argparse.ArgumentParser(description="Sub-clusteriza un cluster aislado")
    ap.add_argument("ply")
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--min-points", type=int, default=20)
    ap.add_argument("--descartar", default=None,
                    help="ids de sub-clusters a botar (modo limpiar)")
    ap.add_argument("--out", default=None,
                    help="ruta de salida (modo limpiar); por defecto <ply>_limpio.ply")
    ap.add_argument("--out-dir", default=None,
                    help="carpeta donde volcar los sub-clusters (modo explorar)")
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(args.ply)
    pts = np.asarray(pcd.points)
    cl = np.array(pcd.cluster_dbscan(eps=args.eps, min_points=args.min_points))
    nclu = int(cl.max()) + 1
    descart = _ids(args.descartar)

    if not descart:
        # Modo explorar: tabla + volcado por sub-cluster
        print(f"{args.ply}: {len(pts):,} puntos | {nclu} sub-clusters "
              f"(ruido {int((cl == -1).sum()):,})")
        tam = sorted(((c, int((cl == c).sum())) for c in range(nclu)),
                     key=lambda t: -t[1])
        print(f"  {'subid':>5} {'n_pts':>8}")
        for c, n in tam:
            print(f"  {c:>5} {n:>8,}")
        if args.out_dir:
            sdir = Path(args.out_dir)
        else:
            base = Path(args.ply).with_suffix("")
            sdir = base.parent / f"{base.name}_sub"
        if sdir.exists():
            for viejo in sdir.glob("subcluster_*.ply"):
                viejo.unlink()
        sdir.mkdir(parents=True, exist_ok=True)
        colors = plt.get_cmap("tab20")(cl % 20)[:, :3]
        colors[cl == -1] = 0.15
        col = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        col.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(sdir / "subclusters_coloreado.ply"), col)
        for c in range(nclu):
            sub = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[cl == c]))
            sub.paint_uniform_color([0.9, 0.3, 0.1])
            o3d.io.write_point_cloud(str(sdir / f"subcluster_{c}.ply"), sub)
        print(f"\n{nclu} sub-clusters -> {sdir}\\")
        print("Identifica los muebles y reejecuta con --descartar <subids> --out <ruta>")
        return

    # Modo limpiar: conservar todo menos los sub-clusters de mueble (el ruido se conserva)
    keep = ~np.isin(cl, list(descart))
    out = Path(args.out) if args.out else Path(args.ply).with_name(
        Path(args.ply).stem + "_limpio.ply")
    limpio = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[keep]))
    if pcd.has_colors():
        limpio.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[keep])
    o3d.io.write_point_cloud(str(out), limpio)
    print(f"Sub-clusters botados: {sorted(descart)}")
    print(f"{int(keep.sum()):,} de {len(pts):,} puntos conservados -> {out}")


if __name__ == "__main__":
    main()
