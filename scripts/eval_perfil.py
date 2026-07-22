"""
eval_perfil.py — Compara segment_foster (baseline) vs segment_by_profile sobre la
nube sintética de 6 clases, con métricas P/R/F1/IoU por clase.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.modules.segmentation import (  # noqa: E402
    segment_by_profile, segment_foster, save_segments)
from app.modules.segmentation_metrics import evaluate, format_report  # noqa: E402
from tests.synthetic_cloud import make_synthetic_foster_6clases  # noqa: E402

NOMBRES = {0: "interior", 1: "suelo", 2: "tambor", 3: "cupula",
           4: "cornisa", 5: "contrafuerte"}


def main():
    pts, truth = make_synthetic_foster_6clases(seed=7)
    print(f"Nube sintética 6 clases: {len(pts):,} puntos\n")

    print("===== BASELINE (segment_foster, 4 clases) =====")
    base = segment_foster(pts)
    mb = evaluate(base.labels, truth, n_classes=6)
    print(format_report(mb, NOMBRES))

    print("\n===== PERFIL (segment_by_profile, 6 clases) =====")
    prof = segment_by_profile(pts)
    mp = evaluate(prof.labels, truth, n_classes=6)
    print(format_report(mp, NOMBRES))

    print("\n===== DELTA IoU (perfil - baseline) =====")
    for c in range(6):
        print(f"  {NOMBRES[c]:13s}: {mb.per_class[c].iou:.3f} -> "
              f"{mp.per_class[c].iou:.3f} ({mp.per_class[c].iou - mb.per_class[c].iou:+.3f})")

    save_segments(pts, prof, Path("output/eval_perfil"))
    print("\ncompleto.ply en output/eval_perfil")


if __name__ == "__main__":
    main()
