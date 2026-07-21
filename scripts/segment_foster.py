"""
segment_foster.py — Segmenta la nube (ROI) del Observatorio y mide sus dimensiones.

Uso (venv activo, desde la raíz del proyecto):
    python scripts/segment_foster.py output/cloud/foster_roi.ply
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.modules.segmentation import (  # noqa: E402
    SegmentationConfig,
    save_segments,
    segment_foster,
)

def main() -> None:
    ap = argparse.ArgumentParser(description="Segmentacion RANSAC del Observatorio")
    ap.add_argument("nube", help="ruta a la nube ROI (.ply/.pcd)")
    ap.add_argument("--out", default="output/segmentation")
    ap.add_argument("--eps-plano", type=float, default=0.05,help="tolerancia del plano de suelo [m] (default 0.05)")
    ap.add_argument("--eps-circulo", type=float, default=0.05,help="tolerancia del circulo del tambor [m] (default 0.05)")
    ap.add_argument("--eps-esfera", type=float, default=0.08,help="tolerancia de la esfera de la cupula [m] (default 0.08)")
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(args.nube)
    pts = np.asarray(pcd.points)
    print(f"Nube: {len(pts):,} puntos")

    config = SegmentationConfig(
        eps_plano=args.eps_plano,
        eps_circulo=args.eps_circulo,
        eps_esfera=args.eps_esfera,
    )
    res = segment_foster(pts, config)
    for valor, nombre in ((0, "resto"), (1, "suelo"), (2, "tambor"), (3, "cupula")):
        print(f"  {nombre:7s}: {(res.labels == valor).sum():,} puntos")

    p = res.params
    print("\nParametros medidos:")
    print(f"  z_suelo      = {p.z_suelo:8.3f} m")
    print(f"  centro       = ({p.cx:.3f}, {p.cy:.3f})")
    print(f"  r_tambor_ext = {p.r_tambor_ext:8.3f} m")
    if p.r_tambor_int is not None:
        print(f"  r_tambor_int = {p.r_tambor_int:8.3f} m"
              f"  -> espesor muro = {p.r_tambor_ext - p.r_tambor_int:.3f} m")
    print(f"  z_top_muro   = {p.z_top_muro:8.3f} m")
    print(f"  r_cupula     = {p.r_cupula:8.3f} m")

    save_segments(pts, res, Path(args.out))
    print(f"\nSegmentos y parametros_foster.json escritos en {args.out}")


if __name__ == "__main__":
    main()
