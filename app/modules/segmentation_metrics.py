"""
segmentation_metrics.py — Métricas de calidad de segmentación por clase.

Precision, recall, F1 e IoU por etiqueta + matriz de confusión. Lógica pura
(sin Qt, sin I/O), para evaluar contra una verdad de referencia conocida.
Definiciones estándar (Salamanca et al. 2024; benchmark ArCH).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    iou: float
    support: int          # nº de puntos cuya verdad es esta clase


@dataclass(frozen=True)
class SegMetrics:
    per_class: dict[int, ClassMetrics]
    macro_f1: float
    macro_iou: float
    confusion: np.ndarray


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def evaluate(pred_labels, true_labels, n_classes: int = 4) -> SegMetrics:
    pred = np.asarray(pred_labels).astype(int)
    true = np.asarray(true_labels).astype(int)
    if pred.shape != true.shape:
        raise ValueError(
            f"pred {pred.shape} y true {true.shape} deben tener igual forma."
        )

    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true, pred):
        confusion[t, p] += 1

    per_class: dict[int, ClassMetrics] = {}
    f1s, ious = [], []
    for c in range(n_classes):
        tp = int(confusion[c, c])
        fp = int(confusion[:, c].sum() - tp)
        fn = int(confusion[c, :].sum() - tp)
        support = int(confusion[c, :].sum())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        iou = _safe_div(tp, tp + fp + fn)
        per_class[c] = ClassMetrics(precision, recall, f1, iou, support)
        if support > 0:                 # macro solo sobre clases presentes
            f1s.append(f1)
            ious.append(iou)

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    macro_iou = float(np.mean(ious)) if ious else 0.0
    return SegMetrics(per_class, macro_f1, macro_iou, confusion)


def format_report(m: SegMetrics, nombres: dict[int, str]) -> str:
    lines = [f"{'clase':10s} {'P':>7s} {'R':>7s} {'F1':>7s} {'IoU':>7s} {'n':>8s}"]
    for c, cm in sorted(m.per_class.items()):
        lines.append(
            f"{nombres.get(c, str(c)):10s} "
            f"{cm.precision:7.3f} {cm.recall:7.3f} {cm.f1:7.3f} {cm.iou:7.3f} "
            f"{cm.support:8,d}"
        )
    lines.append(f"{'macro':10s} {'':7s} {'':7s} "
                 f"{m.macro_f1:7.3f} {m.macro_iou:7.3f}")
    return "\n".join(lines)
