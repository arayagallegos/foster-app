"""
comparar_reconstruccion.py — Poisson vs Ball Pivoting, midiendo AMBAS direcciones.

Leccion aprendida: medir solo "puntos -> malla" (cobertura) es enganoso. Poisson
da 1.75 cm de cobertura pero fabrica casi la mitad de la superficie. Hay que medir
tambien "malla -> puntos" (invencion).

  cobertura  : distancia de cada PUNTO real a la malla. Baja = no falta nada.
  invencion  : distancia de cada punto de la MALLA al punto real mas cercano.
               Alta = la malla tiene superficie donde no habia datos.
  %inventado : fraccion de la superficie a mas de 5 cm de todo punto real.

Uso:
    python cgal_bridge/comparar_reconstruccion.py compuertas
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parent
sys.path.insert(0, str(AQUI))
import cgal_bridge  # noqa: E402


def metricas(pts, mesh, n=200_000):
    if len(mesh.triangles) == 0:
        return None
    sup = np.asarray(mesh.sample_points_uniformly(n).points)
    cob, _ = cKDTree(sup).query(pts)          # puntos -> malla
    inv, _ = cKDTree(pts).query(sup)          # malla  -> puntos
    return dict(cob=cob.mean(), cob95=np.percentile(cob, 95),
                inv=inv.mean(), inv95=np.percentile(inv, 95),
                pct=100 * (inv > 0.05).mean())


def calidad(mesh):
    import pyvista as pv
    F = np.asarray(mesh.triangles)
    V = np.asarray(mesh.vertices)
    caras = np.hstack([np.full((len(F), 1), 3, np.int64), F.astype(np.int64)]).ravel()
    m = pv.PolyData(V, caras)
    b = m.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                manifold_edges=False, non_manifold_edges=False)
    nm = m.extract_feature_edges(boundary_edges=False, feature_edges=False,
                                 manifold_edges=False, non_manifold_edges=True)
    return b.n_cells, nm.n_cells


def fila(nom, mesh, pts, dt):
    m = metricas(pts, mesh)
    if m is None:
        print(f"{nom:22s} | malla vacia"); return
    b, nm = calidad(mesh)
    print(f"{nom:22s} | {len(mesh.vertices):>7,} {len(mesh.triangles):>8,} | "
          f"{m['cob']:>7.4f} | {m['inv']:>7.4f} {m['inv95']:>7.3f} | "
          f"{m['pct']:>6.1f}% | {b:>6} {nm:>5} | {dt:>5.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entidad")
    ap.add_argument("--dir", default="output/refine_real")
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(str(ROOT / args.dir / f"{args.entidad}.ply"))
    pts = np.asarray(pcd.points)
    print(f"{args.entidad}: {len(pts):,} puntos")

    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)

    # espaciado medio entre puntos -> radios para ball pivoting
    dmed = np.mean(pcd.compute_nearest_neighbor_distance())
    print(f"espaciado medio entre puntos: {dmed:.4f} m\n")

    print(f"{'metodo':22s} | {'V':>7} {'F':>8} | {'cobert':>7} | "
          f"{'invent':>7} {'inv95':>7} | {'%>5cm':>7} | {'hoyos':>6} {'nomf':>5} | tiempo")
    print("-" * 108)

    out = ROOT / "output/mallas/comparacion"
    out.mkdir(parents=True, exist_ok=True)

    # --- Poisson (referencia) ---
    t0 = time.time()
    pois, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=11)
    pois.remove_vertices_by_mask(np.asarray(dens) < np.quantile(np.asarray(dens), 0.02))
    fila("Poisson d11", pois, pts, time.time() - t0)
    o3d.io.write_triangle_mesh(str(out / f"{args.entidad}_poisson.ply"), pois)

    # --- Ball pivoting: varios juegos de radios ---
    for mult in ([1.5, 3.0], [2.0, 4.0, 8.0], [1.0, 2.0, 4.0]):
        radios = [dmed * k for k in mult]
        t0 = time.time()
        bpa = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radios))
        dt = time.time() - t0
        fila(f"BPA x{mult}", bpa, pts, dt)
        o3d.io.write_triangle_mesh(
            str(out / f"{args.entidad}_bpa_{'_'.join(str(k) for k in mult)}.ply"), bpa)

    print(f"\nMallas en {out}")
    print("Objetivo: 'invent' y '%>5cm' BAJOS = malla fiel (no fabrica geometria).")


if __name__ == "__main__":
    main()
