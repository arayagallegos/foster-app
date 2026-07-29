"""
barrido_malla.py — Busca los mejores parametros de mallado midiendo CALIDAD y FIDELIDAD.

El problema de iterar "a ojo" es que no hay metrica de fidelidad. Aqui se mide:

  FIDELIDAD  = distancia de cada punto ORIGINAL a la superficie de la malla.
               (media y p95, en metros). Baja = la malla sigue los datos reales.
  CALIDAD    = bordes de hoyo y aristas no-manifold tras reparar con CGAL.
               0 y 0 = malla cerrada y bien formada.

Parametros que se barren:
  depth     : resolucion del octree de Poisson (mas alto = mas detalle)
  quantile  : recorte por densidad (quita geometria inventada)
  radio_n   : radio para estimar normales (chico = conserva detalle fino)

Uso:
    python cgal_bridge/barrido_malla.py compuertas
    python cgal_bridge/barrido_malla.py compuertas --rapido
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

# (depth, quantile, radio_normales)
COMBOS = [
    (9,  0.03, 0.15),
    (11, 0.02, 0.15),
    (11, 0.02, 0.06),
    (12, 0.01, 0.06),
    (12, 0.02, 0.06),
    (13, 0.01, 0.04),
]
COMBOS_RAPIDO = COMBOS[:3]


def calidad(V, F):
    """bordes de hoyo y aristas no-manifold, medidos sobre la malla tal cual."""
    import pyvista as pv
    caras = np.hstack([np.full((len(F), 1), 3, np.int64), F.astype(np.int64)]).ravel()
    m = pv.PolyData(V, caras)
    b = m.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                manifold_edges=False, non_manifold_edges=False)
    nm = m.extract_feature_edges(boundary_edges=False, feature_edges=False,
                                 manifold_edges=False, non_manifold_edges=True)
    return b.n_cells, nm.n_cells


def fidelidad(pts, V, F, n_muestra=200_000):
    """distancia de los puntos ORIGINALES a la superficie de la malla."""
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),
                                  o3d.utility.Vector3iVector(F))
    sup = np.asarray(m.sample_points_uniformly(n_muestra).points)
    d, _ = cKDTree(sup).query(pts)
    return float(d.mean()), float(np.percentile(d, 95))


def una(pts, pcd, depth, q, radio):
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radio, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth)
    if q > 0:
        mesh.remove_vertices_by_mask(np.asarray(dens) < np.quantile(np.asarray(dens), q))
    V = np.asarray(mesh.vertices, np.float64)
    F = np.asarray(mesh.triangles, np.int32)
    V2, F2 = cgal_bridge.fill_holes(V, F, fair=True)
    b, nm = calidad(V2, F2)
    med, p95 = fidelidad(pts, V2, F2)
    return V2, F2, b, nm, med, p95


def main():
    ap = argparse.ArgumentParser(description="Barrido de parametros de mallado")
    ap.add_argument("entidad")
    ap.add_argument("--dir", default="output/refine_real")
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(str(ROOT / args.dir / f"{args.entidad}.ply"))
    pts = np.asarray(pcd.points)
    print(f"{args.entidad}: {len(pts):,} puntos\n")

    combos = COMBOS_RAPIDO if args.rapido else COMBOS
    print(f"{'depth':>5} {'quant':>6} {'rad_n':>6} | {'V':>8} {'F':>8} | "
          f"{'hoyos':>6} {'nomanif':>8} | {'fid_med':>8} {'fid_p95':>8} | {'s':>5}")
    print("-" * 92)

    out = ROOT / "output/mallas/barrido"
    out.mkdir(parents=True, exist_ok=True)
    mejores = []
    for depth, q, radio in combos:
        t0 = time.time()
        try:
            V2, F2, b, nm, med, p95 = una(pts, pcd, depth, q, radio)
        except Exception as e:
            print(f"{depth:>5} {q:>6.2f} {radio:>6.2f} | ERROR: {str(e)[:50]}")
            continue
        dt = time.time() - t0
        ok = "OK" if (b == 0 and nm == 0) else ""
        print(f"{depth:>5} {q:>6.2f} {radio:>6.2f} | {len(V2):>8,} {len(F2):>8,} | "
              f"{b:>6} {nm:>8} | {med:>8.4f} {p95:>8.4f} | {dt:>5.0f} {ok}")
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V2),
                                      o3d.utility.Vector3iVector(F2))
        m.compute_vertex_normals()
        nom = f"{args.entidad}_d{depth}_q{q}_r{radio}.ply"
        o3d.io.write_triangle_mesh(str(out / nom), m)
        mejores.append((med, b + nm, nom))

    print("\nMejores por fidelidad (con malla cerrada y manifold):")
    cerradas = [x for x in mejores if x[1] == 0]
    for med, _, nom in sorted(cerradas)[:3]:
        print(f"  fid_med={med:.4f} m  ->  {nom}")
    if not cerradas:
        print("  (ninguna quedo cerrada+manifold; mira la tabla)")
    print(f"\nMallas en {out}")


if __name__ == "__main__":
    main()
