"""
espejo_zona.py — Reparacion por simetria con ZONA acotada.

Mejora sobre el espejo simple: la simetria no siempre vale en TODA la estructura,
asi que el usuario delimita una CAJA (zona de interes). Solo se reflejan los
puntos de esa zona, y solo se aceptan los reflejados que caen dentro de ella.
Asi se rellena, por ejemplo, un riel mal escaneado sin tocar el resto.

Controles:
  - plano (flecha/bola) : plano de simetria
  - caja                : zona donde aplica la simetria
  - r : refinar el plano automaticamente (maximiza la concordancia EN LA ZONA)
  - t : usar toda la nube (ignorar la caja) / volver a la zona
  - a : APLICAR: los puntos verdes se integran a la nube. Permite mover el plano
        y la caja y seguir reparando otra zona sobre el resultado acumulado.
  - u : deshacer la ultima aplicacion
  - s : guardar nube reparada -> output/reparadas/<entidad>_simetria.ply
  - q : salir

Metricas en pantalla:
  CONCORDANCIA : % de reflejados que caen sobre puntos reales -> valida la simetria
  rellenarian  : reflejados que caen en zona vacia -> los que aportan

Uso:
    python cgal_bridge/espejo_zona.py compuertas
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pyvista as pv
from scipy.optimize import minimize
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
TOL = 0.05


def params_de(n, o):
    n = np.asarray(n, float); n /= np.linalg.norm(n)
    return np.array([np.arctan2(n[1], n[0]), np.arccos(np.clip(n[2], -1, 1)), o @ n])


def plano_de(q):
    th, ph, d = q
    n = np.array([np.sin(ph)*np.cos(th), np.sin(ph)*np.sin(th), np.cos(ph)])
    return n, n * d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entidad")
    ap.add_argument("--dir", default="output/refine_real")
    ap.add_argument("--gap", type=float, default=0.05)
    ap.add_argument("--continuar", action="store_true",
                    help="retomar desde output/reparadas/<entidad>_simetria.ply "
                         "en vez de la nube original")
    args = ap.parse_args()

    reparada = ROOT / f"output/reparadas/{args.entidad}_simetria.ply"
    if args.continuar and reparada.exists():
        ruta = reparada
        print("continuando desde la nube YA REPARADA")
    else:
        ruta = ROOT / args.dir / f"{args.entidad}.ply"
    pts = np.asarray(o3d.io.read_point_cloud(str(ruta)).points)
    print(f"{args.entidad}: {len(pts):,} puntos  ({ruta.name})")

    mn, mx = pts.min(0), pts.max(0)
    # `pts` y `tree` son la nube DE TRABAJO: crecen con cada aplicacion ('a'),
    # asi se pueden encadenar varios planos de simetria sobre el resultado.
    S = {"q": None, "caja": (mn, mx), "usar_caja": True, "nuevos": np.empty((0, 3)),
         "pts": pts, "tree": cKDTree(pts), "hist": [], "n0": len(pts)}

    def en_caja(p):
        if not S["usar_caja"]:
            return np.ones(len(p), bool)
        a, b = S["caja"]
        return np.all((p >= a) & (p <= b), axis=1)

    def concordancia(q, sub, tsub):
        n, o = plano_de(q)
        r = sub - 2.0 * np.outer((sub - o) @ n, n)
        d, _ = tsub.query(r)
        return float((d < TOL).mean())

    pl = pv.Plotter()
    pl.set_background("black")
    pl.add_mesh(pv.PolyData(pts), color="gray", point_size=2.0, name="real")

    def actualizar(q, etq=""):
        S["q"] = q
        p = S["pts"]
        n, o = plano_de(q)
        origen = p[en_caja(p)]                    # solo se refleja la zona
        refl = origen - 2.0 * np.outer((origen - o) @ n, n)
        refl = refl[en_caja(refl)]                # y solo se acepta en la zona
        d, _ = S["tree"].query(refl) if len(refl) else (np.array([]), None)
        conc = float((d < TOL).mean()) if len(d) else 0.0
        nuevos = refl[d > args.gap] if len(d) else refl
        S["nuevos"] = nuevos
        if len(nuevos):
            pl.add_mesh(pv.PolyData(nuevos), color="lime", point_size=3.0,
                        name="nuevos")
        else:
            pl.remove_actor("nuevos")
        agregados = len(p) - S["n0"]
        pl.add_text(f"CONCORDANCIA: {100*conc:.1f}%   (maximizar)\n"
                    f"rellenarian: {len(nuevos):,}   "
                    f"acumulados: {agregados:,} ({len(S['hist'])} planos)\n"
                    f"zona: {'CAJA' if S['usar_caja'] else 'TODA la nube'}   {etq}",
                    name="info", position="upper_left", font_size=11, color="white")
        print(f"conc={100*conc:5.1f}%  nuevos={len(nuevos):,}  "
              f"acumulados={agregados:,}  {etq}")

    def on_plane(normal, origin):
        actualizar(params_de(normal, np.asarray(origin, float)))

    def on_box(box_widget):
        b = box_widget.bounds
        S["caja"] = (np.array([b[0], b[2], b[4]]), np.array([b[1], b[3], b[5]]))
        if S["q"] is not None:
            actualizar(S["q"])

    def aplicar():
        """Integra los puntos verdes a la nube de trabajo, para encadenar planos."""
        if not len(S["nuevos"]):
            print("nada que aplicar"); return
        S["hist"].append(len(S["pts"]))
        S["pts"] = np.vstack([S["pts"], S["nuevos"]])
        S["tree"] = cKDTree(S["pts"])
        print(f"[aplicado] +{len(S['nuevos']):,} -> nube de trabajo: {len(S['pts']):,}")
        S["nuevos"] = np.empty((0, 3))
        pl.remove_actor("nuevos")
        pl.add_mesh(pv.PolyData(S["pts"]), color="gray", point_size=2.0, name="real")
        if S["q"] is not None:
            actualizar(S["q"], "(aplicado)")

    def deshacer():
        if not S["hist"]:
            print("nada que deshacer"); return
        n = S["hist"].pop()
        S["pts"] = S["pts"][:n]
        S["tree"] = cKDTree(S["pts"])
        pl.add_mesh(pv.PolyData(S["pts"]), color="gray", point_size=2.0, name="real")
        print(f"[deshecho] nube de trabajo: {len(S['pts']):,}")
        if S["q"] is not None:
            actualizar(S["q"], "(deshecho)")

    def refinar():
        if S["q"] is None:
            return
        sub = S["pts"][en_caja(S["pts"])]
        if len(sub) > 12000:
            sub = sub[np.random.default_rng(0).choice(len(sub), 12000, replace=False)]
        tsub = cKDTree(sub)
        print("refinando...")
        res = minimize(lambda q: -concordancia(q, sub, tsub), S["q"],
                       method="Nelder-Mead",
                       options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-4})
        actualizar(res.x, "(refinado)")

    def alternar():
        S["usar_caja"] = not S["usar_caja"]
        if S["q"] is not None:
            actualizar(S["q"])

    def guardar():
        # Guarda la nube de TRABAJO (todo lo acumulado) + lo verde pendiente.
        todos = S["pts"]
        if len(S["nuevos"]):
            todos = np.vstack([todos, S["nuevos"]])
        if len(todos) == S["n0"]:
            print("nada reparado que guardar"); return
        out = ROOT / "output/reparadas"
        out.mkdir(parents=True, exist_ok=True)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(todos))
        ruta = out / f"{args.entidad}_simetria.ply"
        o3d.io.write_point_cloud(str(ruta), pcd)
        print(f"guardado: {S['n0']:,} originales + {len(todos)-S['n0']:,} "
              f"reparados = {len(todos):,} -> {ruta}")

    pl.add_plane_widget(on_plane, normal="y", origin=pts.mean(0), implicit=True)
    pl.add_box_widget(on_box, bounds=[mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]],
                      rotation_enabled=False, color="cyan")
    for k, f in (("r", refinar), ("t", alternar), ("a", aplicar),
                 ("u", deshacer), ("s", guardar)):
        pl.add_key_event(k, f)
    print(__doc__)
    pl.show()


if __name__ == "__main__":
    main()
