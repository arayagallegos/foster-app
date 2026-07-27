"""
refine_interior_real.py — Refina la clase interior de la segmentación por perfil
del Foster con DBSCAN. Flujo asistido en dos pasos:

  1) Sin --compuertas: corre segment_by_profile + DBSCAN sobre el interior, imprime
     la tabla de clusters (id, n_pts, frac_en_cascara, planaridad) y escribe la nube
     coloreada por cluster para que elijas a ojo el id de las compuertas.
  2) Con --compuertas <id>: escribe el etiquetado final de 8 clases. Con --eliminar,
     además escribe estructural.ply (sin la clase descartado 6).

Uso (venv activo, desde la raíz):
    python scripts/refine_interior_real.py <nube.ply>
    python scripts/refine_interior_real.py <nube.ply> --compuertas 2 --eliminar
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.modules.segmentation import (  # noqa: E402
    ProfileConfig, SegmentationResult, save_segments, segment_by_profile)
from app.modules.refine_interior import (  # noqa: E402
    INTERIOR, RefineConfig, dbscan_interior, eliminar_descartados, info_clusters,
    reetiquetar_interior_por_clusters)

NOMBRES = {0: "interior", 1: "suelo", 2: "tambor", 3: "cupula", 4: "cornisa",
           5: "contrafuerte", 6: "descartado", 7: "compuertas"}


def _ids(texto: str | None) -> tuple[int, ...]:
    """'9,11, 20' -> (9, 11, 20). Vacío/None -> ()."""
    if not texto:
        return ()
    return tuple(int(x) for x in texto.replace(" ", "").split(",") if x != "")


def main():
    ap = argparse.ArgumentParser(description="Refina el interior con DBSCAN")
    ap.add_argument("nube")
    ap.add_argument("--out", default="output/refine_real")
    ap.add_argument("--compuertas", type=int, default=None,
                    help="id del cluster de compuertas (de la tabla del paso 1) -> clase 7")
    ap.add_argument("--descartar", default=None,
                    help="ids de clusters a botar (telescopio/equipo), ej. 9,11,20 -> clase 6")
    ap.add_argument("--cupula", default=None,
                    help="ids de clusters de vigas a mandar a cupula, ej. 1,5 -> clase 3")
    ap.add_argument("--suelo", default=None,
                    help="ids de clusters de piso mal clasificado, ej. 14,15 -> clase 1")
    ap.add_argument("--eliminar", action="store_true",
                    help="escribir tambien estructural.ply sin la clase descartado")
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--min-points", type=int, default=20)
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(args.nube)
    pts = np.asarray(pcd.points)
    print(f"Nube: {len(pts):,} puntos")

    seg = segment_by_profile(pts, ProfileConfig())
    cfg = RefineConfig(eps=args.eps, min_points=args.min_points)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    id_comp = args.compuertas
    ids_desc = _ids(args.descartar)
    ids_cup = _ids(args.cupula)
    ids_sue = _ids(args.suelo)
    modo_etiquetado = id_comp is not None or ids_desc or ids_cup or ids_sue

    if not modo_etiquetado:
        # Paso 1: exploración — clusteriza el interior, muestra la tabla y vuelca
        # los clusters grandes como archivos separados para identificarlos a ojo.
        idx = np.where(seg.labels == INTERIOR)[0]
        pts_int = pts[idx]
        cl = dbscan_interior(pts_int, cfg.eps, cfg.min_points)
        infos = info_clusters(pts_int, cl, seg.params.centro_cupula,
                              seg.params.r_cupula, cfg.margen_cascara)
        infos.sort(key=lambda i: -i.n_pts)
        print(f"\nInterior: {len(pts_int):,} puntos | {len(infos)} clusters "
              f"(+ ruido {int((cl == -1).sum()):,})")
        print(f"  {'id':>4} {'n_pts':>9} {'frac_cascara':>12} {'planaridad':>11}")
        for i in infos:
            print(f"  {i.id:>4} {i.n_pts:>9,} {i.frac_en_cascara:>12.2f} "
                  f"{i.planaridad:>11.2f}")

        colors = plt.get_cmap("tab20")(cl % 20)[:, :3]
        colors[cl == -1] = 0.15
        pint = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_int))
        pint.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(out / "interior_clusters.ply"), pint)

        # Un .ply por CADA cluster (color plano), para cargarlos de a uno y ver
        # sin ambigüedad qué es cada id. El ruido de DBSCAN queda aparte.
        cdir = out / "clusters"
        if cdir.exists():
            for viejo in cdir.glob("cluster_*.ply"):
                viejo.unlink()                       # limpia volcados previos
        cdir.mkdir(parents=True, exist_ok=True)
        for i in infos:
            sub = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_int[cl == i.id]))
            sub.paint_uniform_color([0.9, 0.3, 0.1])
            o3d.io.write_point_cloud(str(cdir / f"cluster_{i.id}.ply"), sub)
        ruido = pts_int[cl == -1]
        if len(ruido):
            pr = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ruido))
            pr.paint_uniform_color([0.5, 0.5, 0.5])
            o3d.io.write_point_cloud(str(cdir / "ruido.ply"), pr)

        print(f"\nColoreado -> {out/'interior_clusters.ply'}")
        print(f"{len(infos)} clusters individuales + ruido.ply -> {cdir}\\")
        print("Identifica los ids y reejecuta, por ej.:")
        print("  --compuertas <id> --descartar 9,11 --eliminar")
        return

    # Paso 2: etiquetado por clusters elegidos a mano
    labels = reetiquetar_interior_por_clusters(
        pts, seg.labels, cfg, id_compuertas=id_comp,
        ids_descartar=ids_desc, ids_cupula=ids_cup, ids_suelo=ids_sue)
    print(f"compuertas={id_comp}  descartar={len(ids_desc)} ids  "
          f"cupula={len(ids_cup)} ids  suelo={len(ids_sue)} ids")
    for v, n in NOMBRES.items():
        print(f"  {n:13s}: {(labels == v).sum():,}")
    save_segments(pts, SegmentationResult(labels=labels, params=seg.params), out)
    print(f"\ncompleto.ply en {out}")

    if args.eliminar:
        pe, le = eliminar_descartados(pts, labels)
        pcd_e = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pe))
        o3d.io.write_point_cloud(str(out / "estructural.ply"), pcd_e)
        print(f"estructural.ply ({len(pe):,} puntos, sin descartado) en {out}")


if __name__ == "__main__":
    main()
