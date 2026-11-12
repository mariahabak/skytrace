"""Classification metrics computed with tensor operations.

Author: Mariah Habak
"""

import torch

CLASS_NAMES = ("bird", "drone")


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor,
                     n_classes: int = 2) -> torch.Tensor:
    """(n, n) counts; rows are true classes, columns are predictions."""
    index = y_true.long() * n_classes + y_pred.long()
    return torch.bincount(index, minlength=n_classes * n_classes).view(n_classes, n_classes)


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    if len(y_true) == 0:
        return float("nan")
    return (y_true == y_pred).float().mean().item()


def per_class(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    m = confusion_matrix(y_true, y_pred).float()
    tp = m.diagonal()
    predicted = m.sum(dim=0)
    actual = m.sum(dim=1)
    precision = torch.where(predicted > 0, tp / predicted.clamp(min=1), torch.zeros_like(tp))
    recall = torch.where(actual > 0, tp / actual.clamp(min=1), torch.zeros_like(tp))
    denom = precision + recall
    f1 = torch.where(denom > 0, 2 * precision * recall / denom.clamp(min=1e-12),
                     torch.zeros_like(tp))
    return {CLASS_NAMES[c]: {"precision": precision[c].item(), "recall": recall[c].item(),
                             "f1": f1[c].item(), "support": int(actual[c].item())}
            for c in range(len(CLASS_NAMES))}


def macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    if len(y_true) == 0:
        return float("nan")
    scores = per_class(y_true, y_pred)
    return sum(s["f1"] for s in scores.values()) / len(scores)


def summary(y_true: torch.Tensor, scores: torch.Tensor) -> dict:
    y_pred = scores.argmax(dim=1)
    return {"accuracy": accuracy(y_true, y_pred), "macro_f1": macro_f1(y_true, y_pred),
            "per_class": per_class(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "n": int(len(y_true))}
