"""
separar_caras.py — Separa una entidad en sus dos CARAS (interior / exterior) y
malla cada una por separado.

Motivacion: el escaneo capturo ambas caras de los elementos (el panel de las
compuertas tiene ~3 cm de espesor). Al mallar todo junto, Advancing Front
reconstruye las dos hojas y en zonas ralas se cruzan -> auto-intersecciones y
huecos aparentes. Separandolas se obtienen dos superficies limpias, y el
ESPESOR queda medido como dato derivado (no hay que asignarlo).

Criterio de separacion: para cada punto se compara su distancia al centro de la
cupula contra el PROMEDIO LOCAL de sus vecinos. Mas lejos que sus vecinos ->
cara exterior; mas cerca -> cara interior.

Es una referencia LOCAL a proposito: la global (distancia al centro a secas) no
sirve porque el panel abarca ~1 m de rango radial por su propia curvatura, y los
pocos centimetros de espesor quedan sepultados. Restar el promedio local elimina
la curvatura y deja solo la separacion entre las dos laminas.

(Un intento previo usando la ORIENTACION de las normales fallo: la propagacion de
`orient_normals_consistent_tangent_plane` se voltea al cruzar las hendiduras de
los rieles, y el resultado partia el panel en franjas en vez de en caras.)

Uso:
    python cgal_bridge/separar_caras.py compuertas
    python cgal_bridge/separar_caras.py cupula --radio-normales 0.10
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parent
sys.path.insert(0, str(AQUI))
import cgal_bridge  # noqa: E402


def calidad(V, F):
    import pyvista as pv
    caras = np.hstack([np.full((len(F), 1), 3, np.int64), F.astype(np.int64)]).ravel()
    m = pv.PolyData(V, caras)
    b = m.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                manifold_edges=False, non_manifold_edges=False)
    nm = m.extract_feature_edges(boundary_edges=False, feature_edges=False,
                                 manifold_edges=False, non_manifold_edges=True)
    return b.n_cells, nm.n_cells


def mallar(P, nom, out):
    d = float(np.mean(cKDTree(P).query(P, k=2)[0][:, 1]))
    V, F = cgal_bridge.advancing_front(np.ascontiguousarray(P, dtype=np.float64),
                                       5.0, 0.52, 14.0 * d)
    st = cgal_bridge.mesh_stats(V, F)
    b, nm = calidad(V, F)
    print(f"  {nom:22s} pts={len(P):>7,} F={len(F):>7,} bordes={b:>7,} "
          f"nomanif={nm:>4} auto-int={st['self_intersects']}")
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),
                                  o3d.utility.Vector3iVector(F))
    m.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out / f"{nom}.ply"), m)
    return V, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entidad")
    ap.add_argument("--dir", default="output/refine_real")
    ap.add_argument("--radio-local", type=float, default=0.25,
                    help="radio del vecindario para el promedio local (m). Debe ser "
                         "bastante mayor que el espesor y menor que la curvatura.")
    ap.add_argument("--k-voto", type=int, default=12,
                    help="vecinos usados en el voto mayoritario (coherencia espacial)")
    ap.add_argument("--iter-voto", type=int, default=6,
                    help="iteraciones del voto mayoritario")
    args = ap.parse_args()

    par = json.loads((ROOT / "output/perfil_real/parametros_foster.json").read_text())
    centro = np.array(par["centro_cupula"])

    pcd = o3d.io.read_point_cloud(str(ROOT / args.dir / f"{args.entidad}.ply"))
    P = np.asarray(pcd.points)
    print(f"{args.entidad}: {len(P):,} puntos")

    # --- separacion por distancia radial RELATIVA AL VECINDARIO ---
    d = np.linalg.norm(P - centro, axis=1)
    arbol = cKDTree(P)
    vecinos = arbol.query_ball_point(P, args.radio_local)
    d_local = np.array([d[v].mean() for v in vecinos])
    resid = d - d_local            # + = mas afuera que sus vecinos
    ext = resid > 0

    # --- coherencia espacial: voto mayoritario entre vecinos ---
    # Clasificar punto a punto deja un moteado de errores: los puntos con residuo
    # cercano a cero (justo entre las dos capas) caen a un lado casi al azar, y
    # cada error abre un hoyo en su cara y ensucia la otra. Un punto rodeado de
    # vecinos "exterior" debe ser exterior aunque su propio residuo sea ambiguo.
    _, vk = arbol.query(P, k=args.k_voto + 1)
    for it in range(args.iter_voto):
        votos = ext[vk[:, 1:]].mean(axis=1)      # fraccion de vecinos "exterior"
        nuevo = votos > 0.5
        cambios = int((nuevo != ext).sum())
        ext = nuevo
        print(f"  voto mayoritario iter {it+1}: {cambios:,} puntos reasignados")
        if cambios == 0:
            break
    inte = ~ext
    print(f"  residuo radial local: p10={np.percentile(resid,10):+.4f} "
          f"mediana={np.median(resid):+.4f} p90={np.percentile(resid,90):+.4f} m")
    print(f"  cara exterior: {ext.sum():,} ({100*ext.mean():.0f}%)   "
          f"cara interior: {inte.sum():,} ({100*inte.mean():.0f}%)")

    # ¿bimodal? si las dos caras existen, el residuo tiene dos lobulos
    h, e = np.histogram(resid, bins=40)
    print("  histograma del residuo (dos lobulos = dos caras):")
    for k in range(40):
        if h[k] > 0:
            print(f"    {e[k]:+.3f} {'#' * int(46 * h[k] / h.max())} {h[k]:,}")

    # --- espesor: distancia de cada punto interior a la cara exterior ---
    if ext.sum() > 100 and inte.sum() > 100:
        d, _ = cKDTree(P[ext]).query(P[inte])
        print(f"  ESPESOR medido: mediana={np.median(d):.4f} m  "
              f"p10={np.percentile(d,10):.4f}  p90={np.percentile(d,90):.4f}")

    out = ROOT / "output/mallas/caras"
    out.mkdir(parents=True, exist_ok=True)
    print("\nmallando por separado:")
    mallar(P, f"{args.entidad}_junto", out)
    if ext.sum() > 100:
        mallar(P[ext], f"{args.entidad}_exterior", out)
    if inte.sum() > 100:
        mallar(P[inte], f"{args.entidad}_interior", out)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
