"""tests/test_segmentation_metrics.py — Métricas de segmentación."""
import numpy as np
import pytest

from app.modules.segmentation_metrics import evaluate, format_report


def test_evaluate_perfecto_da_uno():
    y = np.array([0, 1, 2, 3, 1, 2])
    m = evaluate(y, y, n_classes=4)
    for c in (0, 1, 2, 3):
        if m.per_class[c].support > 0:
            assert m.per_class[c].f1 == pytest.approx(1.0)
            assert m.per_class[c].iou == pytest.approx(1.0)
    assert m.macro_f1 == pytest.approx(1.0)
    assert m.macro_iou == pytest.approx(1.0)


def test_evaluate_valores_conocidos():
    # clase 1: verdad=[1,1,1,0], pred=[1,1,0,1]  -> TP=2, FP=1, FN=1
    true = np.array([1, 1, 1, 0])
    pred = np.array([1, 1, 0, 1])
    m = evaluate(pred, true, n_classes=2)
    c1 = m.per_class[1]
    assert c1.precision == pytest.approx(2 / 3)
    assert c1.recall == pytest.approx(2 / 3)
    assert c1.f1 == pytest.approx(2 / 3)
    assert c1.iou == pytest.approx(0.5)          # 2/(2+1+1)
    assert c1.support == 3                        # 3 puntos verdaderos de clase 1


def test_evaluate_clase_ausente_no_rompe():
    true = np.array([0, 0, 0])
    pred = np.array([0, 0, 0])
    m = evaluate(pred, true, n_classes=4)
    # clases 1,2,3 sin support: no deben aportar NaN al macro
    assert not np.isnan(m.macro_f1)
    assert m.per_class[2].support == 0


def test_confusion_shape_y_conteo():
    true = np.array([1, 1, 2])
    pred = np.array([1, 2, 2])
    m = evaluate(pred, true, n_classes=4)
    assert m.confusion.shape == (4, 4)
    assert m.confusion[1, 1] == 1     # verdad 1, pred 1
    assert m.confusion[1, 2] == 1     # verdad 1, pred 2
    assert m.confusion[2, 2] == 1


def test_format_report_incluye_metricas_y_clases():
    y = np.array([0, 1, 2, 3])
    txt = format_report(evaluate(y, y), {0: "resto", 1: "suelo", 2: "tambor", 3: "cupula"})
    assert "tambor" in txt and "F1" in txt and "IoU" in txt
