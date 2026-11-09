"""Classification metrics, written in NumPy so that they can be tested.

Authors: Maria Habak, [Partner name]
"""

import numpy as np

CLASS_NAMES = ("bird", "drone")


def confusion_matrix(y_true, y_pred, n_classes: int = 2) -> np.ndarray:
    """Rows are true classes, columns are predicted classes."""
    m = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        m[int(t), int(p)] += 1
    return m


def per_class(y_true, y_pred, n_classes: int = 2) -> dict:
    m = confusion_matrix(y_true, y_pred, n_classes)
    out = {}
    for c in range(n_classes):
        tp = m[c, c]
        predicted = m[:, c].sum()
        actual = m[c, :].sum()
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        out[CLASS_NAMES[c]] = {"precision": float(precision),
                               "recall": float(recall), "f1": float(f1),
                               "support": int(actual)}
    return out


def macro_f1(y_true, y_pred, n_classes: int = 2) -> float:
    scores = per_class(y_true, y_pred, n_classes)
    return float(np.mean([s["f1"] for s in scores.values()]))


def summary(y_true, scores: np.ndarray) -> dict:
    y_pred = scores.argmax(axis=1)
    return {"macro_f1": macro_f1(y_true, y_pred),
            "accuracy": float((y_pred == np.asarray(y_true)).mean()),
            "per_class": per_class(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "ece": expected_calibration_error(y_true, scores)}


def expected_calibration_error(y_true, scores: np.ndarray, bins: int = 10) -> float:
    """Average gap between confidence and accuracy, weighted by bin size."""
    confidence = scores.max(axis=1)
    correct = scores.argmax(axis=1) == np.asarray(y_true)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(ece)


def error_coverage(y_true, scores: np.ndarray) -> tuple:
    """Coverage and error when the least confident clips are rejected first.

    Returns arrays (coverage, error): for every k, keep the k most
    confident clips and measure the error among them.
    """
    confidence = scores.max(axis=1)
    wrong = (scores.argmax(axis=1) != np.asarray(y_true)).astype(float)
    order = np.argsort(-confidence, kind="stable")
    kept = np.arange(1, len(order) + 1)
    error = np.cumsum(wrong[order]) / kept
    coverage = kept / len(order)
    return coverage, error


def area_under_error_coverage(y_true, scores: np.ndarray) -> float:
    coverage, error = error_coverage(y_true, scores)
    return float(np.trapezoid(error, coverage)) if hasattr(np, "trapezoid") \
        else float(np.trapz(error, coverage))
