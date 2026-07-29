"""
mallar_entidad.py — Pipeline real: NUBE de una entidad -> MALLA -> REPARACION (CGAL).

Pasos:
  1. Normales (necesarias para Poisson) orientadas de forma consistente.
  2. Reconstruccion de superficie de Poisson (Open3D).
  3. Recorte por densidad: Poisson "inventa" superficie donde no habia puntos;
     se eliminan los vertices de baja densidad para no quedarse con geometria
     alucinada. Es el parametro que controla fidelidad vs cierre.
  4. Diagnostico con CGAL (hoyos / cerrada / auto-intersecciones).
  5. Rellenado de hoyos con CGAL + diagnostico final.

Uso:
    python cgal_bridge/mallar_entidad.py compuertas
    python cgal_bridge/mallar_entidad.py cupula --depth 10 --quantile 0.05
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parent
sys.path.insert(0, str(AQUI))
import cgal_bridge  # noqa: E402


def stats(nom, V, F):
    s = cgal_bridge.mesh_stats(V, F)
    print(f"  {nom:9s} V={s['n_vertices']:>7,} F={s['n_faces']:>7,} "
          f"hoyos={s['n_holes']:>4}  cerrada={str(s['is_closed']):5s} "
          f"auto-int={s['self_intersects']}")
    return s


def main():
    ap = argparse.ArgumentParser(description="Nube de entidad -> malla -> reparacion")
    ap.add_argument("entidad", help="p.ej. compuertas, cupula, tambor")
    ap.add_argument("--dir", default="output/refine_real")
    ap.add_argument("--depth", type=int, default=9,
                    help="profundidad de Poisson (mas alto = mas detalle)")
    ap.add_argument("--quantile", type=float, default=0.03,
                    help="fraccion de vertices de menor densidad a eliminar "
                         "(0 = no recortar; sube si aparece geometria inventada)")
    ap.add_argument("--no-fill", action="store_true", help="no rellenar hoyos")
    args = ap.parse_args()

    ruta = ROOT / args.dir / f"{args.entidad}.ply"
    pcd = o3d.io.read_point_cloud(str(ruta))
    print(f"{args.entidad}: {len(pcd.points):,} puntos")

    # 1. Normales
    t0 = time.time()
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)
    print(f"normales: {time.time()-t0:.1f} s")

    # 2. Poisson
    t0 = time.time()
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=args.depth)
    dens = np.asarray(dens)
    print(f"poisson(depth={args.depth}): {len(mesh.vertices):,} V, "
          f"{len(mesh.triangles):,} F  ({time.time()-t0:.1f} s)")

    # 3. Recorte por densidad
    if args.quantile > 0:
        umbral = np.quantile(dens, args.quantile)
        mesh.remove_vertices_by_mask(dens < umbral)
        print(f"recorte por densidad (q={args.quantile}): "
              f"{len(mesh.vertices):,} V, {len(mesh.triangles):,} F")

    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.triangles, dtype=np.int32)

    out = ROOT / "output/mallas"
    out.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out / f"{args.entidad}_cruda.ply"), mesh)

    # 4-5. Diagnostico + reparacion con CGAL
    print("\nDiagnostico CGAL:")
    stats("cruda", V, F)
    if args.no_fill:
        return
    t0 = time.time()
    V2, F2 = cgal_bridge.fill_holes(V, F, fair=True)
    print(f"  (fill_holes: {time.time()-t0:.1f} s)")
    stats("reparada", V2, F2)

    m2 = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V2), o3d.utility.Vector3iVector(F2))
    m2.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out / f"{args.entidad}_reparada.ply"), m2)
    print(f"\n-> {out}\\{args.entidad}_cruda.ply  y  {args.entidad}_reparada.ply")


if __name__ == "__main__":
    main()
