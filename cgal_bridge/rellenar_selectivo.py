"""
rellenar_selectivo.py — BPA + rellenado SELECTIVO de hoyos con CGAL.

Idea: reconstruir con Ball Pivoting (no inventa geometria) y luego cerrar SOLO
los hoyos pequenos —gaps de muestreo, interpolables honestamente entre puntos
reales vecinos— dejando abiertas las aberturas grandes, que corresponden a zonas
que el escaner no capturo.

Muestra la distribucion de tamanos de hoyo y el efecto de varios umbrales,
midiendo siempre las DOS direcciones (cobertura e invencion).

Uso:
    python cgal_bridge/rellenar_selectivo.py compuertas
    python cgal_bridge/rellenar_selectivo.py compuertas --umbral 0.5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parent
sys.path.insert(0, str(AQUI))
import cgal_bridge  # noqa: E402


def bordes_no_manifold(V, F):
    import pyvista as pv
    caras = np.hstack([np.full((len(F), 1), 3, np.int64), F.astype(np.int64)]).ravel()
    m = pv.PolyData(V, caras)
    b = m.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                manifold_edges=False, non_manifold_edges=False)
    nm = m.extract_feature_edges(boundary_edges=False, feature_edges=False,
                                 manifold_edges=False, non_manifold_edges=True)
    return b.n_cells, nm.n_cells


def metricas(pts, V, F, n=200_000):
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),
                                  o3d.utility.Vector3iVector(F))
    sup = np.asarray(m.sample_points_uniformly(n).points)
    cob, _ = cKDTree(sup).query(pts)
    inv, _ = cKDTree(pts).query(sup)
    return cob.mean(), inv.mean(), 100 * (inv > 0.05).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entidad")
    ap.add_argument("--dir", default="output/refine_real")
    ap.add_argument("--radios", default="2,4,8", help="multiplos del espaciado medio")
    ap.add_argument("--umbral", type=float, default=None,
                    help="perimetro maximo (m) a rellenar; sin esto, barre varios")
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(str(ROOT / args.dir / f"{args.entidad}.ply"))
    pts = np.asarray(pcd.points)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)
    d = float(np.mean(pcd.compute_nearest_neighbor_distance()))
    print(f"{args.entidad}: {len(pts):,} puntos | espaciado medio {d:.4f} m")

    mult = [float(x) for x in args.radios.split(",")]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([d * k for k in mult]))
    V = np.asarray(mesh.vertices, np.float64)
    F = np.asarray(mesh.triangles, np.int32)
    print(f"BPA x{mult}: {len(V):,} V, {len(F):,} F")

    # --- distribucion de tamanos de hoyo ---
    hs = cgal_bridge.hole_sizes(V, F)
    per = np.array([h["perimetro"] for h in hs])
    print(f"\nHoyos: {len(per):,}")
    print(f"  perimetro  mediana={np.median(per):.3f} m  p90={np.percentile(per,90):.3f}  "
          f"max={per.max():.3f}")
    print(f"  {'umbral':>8} {'hoyos<=umbral':>14} {'% del total':>12}")
    for u in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
        n = int((per <= u).sum())
        print(f"  {u:>8.2f} {n:>14,} {100*n/len(per):>11.1f}%")

    umbrales = [args.umbral] if args.umbral else [0.25, 0.5, 1.0, 0.0]
    print(f"\n{'umbral':>8} | {'V':>8} {'F':>8} | {'hoyos':>7} {'nomanif':>8} | "
          f"{'cobert':>7} {'invent':>7} {'%>5cm':>7}")
    print("-" * 76)
    out = ROOT / "output/mallas/selectivo"
    out.mkdir(parents=True, exist_ok=True)
    for u in umbrales:
        V2, F2 = cgal_bridge.fill_holes(V, F, fair=True, max_perimetro=u)
        b, nm = bordes_no_manifold(V2, F2)
        cob, inv, pct = metricas(pts, V2, F2)
        etq = f"{u:.2f}" if u > 0 else "TODOS"
        print(f"{etq:>8} | {len(V2):>8,} {len(F2):>8,} | {b:>7,} {nm:>8} | "
              f"{cob:>7.4f} {inv:>7.4f} {pct:>6.1f}%")
        m2 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V2),
                                       o3d.utility.Vector3iVector(F2))
        m2.compute_vertex_normals()
        o3d.io.write_triangle_mesh(
            str(out / f"{args.entidad}_u{etq}.ply"), m2)
    print(f"\nMallas en {out}")


if __name__ == "__main__":
    main()
