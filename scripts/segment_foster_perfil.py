"""
segment_foster_perfil.py — Corre segment_by_profile sobre la nube real del Observatorio
y escribe la segmentación de 6 clases coloreada.

Uso (venv activo):
    python scripts/segment_foster_perfil.py <nube.ply> [--out output/perfil_real]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.modules.segmentation import (  # noqa: E402
    ProfileConfig, save_segments, segment_by_profile)

NOMBRES = {0: "interior", 1: "suelo", 2: "tambor", 3: "cupula",
           4: "cornisa", 5: "contrafuerte"}


def main():
    ap = argparse.ArgumentParser(description="Segmentacion por perfil (6 clases)")
    ap.add_argument("nube")
    ap.add_argument("--out", default="output/perfil_real")
    ap.add_argument("--h-muro", type=float, default=1.5)
    ap.add_argument("--h-cornisa", type=float, default=0.9)
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(args.nube)
    pts = np.asarray(pcd.points)
    print(f"Nube: {len(pts):,} puntos")

    cfg = ProfileConfig(h_muro=args.h_muro, h_cornisa=args.h_cornisa)
    res = segment_by_profile(pts, cfg)
    for v, n in NOMBRES.items():
        print(f"  {n:13s}: {(res.labels == v).sum():,}")

    save_segments(pts, res, Path(args.out))
    print(f"\ncompleto.ply en {args.out}")


if __name__ == "__main__":
    main()
