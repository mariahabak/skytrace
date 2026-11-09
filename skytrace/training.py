"""A training loop that validates every epoch and keeps the best state.

Authors: Maria Habak, [Partner name]
Reference: PyTorch "Optimizing Model Parameters" tutorial,
https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
"""

import copy
import time

import numpy as np
import torch
from torch import nn

from skytrace.evaluation.metrics import macro_f1


def pick_device(preferred: str = "auto") -> torch.device:
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def class_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=2).float()
    return counts.sum() / (2.0 * counts.clamp(min=1))


class Trainer:
    """Trains ``module`` with Adam and cross-entropy.

    After each epoch it computes the validation loss and macro-F1, keeps
    a copy of the weights with the best validation macro-F1, and stops
    early when that has not improved for ``patience`` epochs.
    """

    def __init__(self, epochs: int, learning_rate: float, weight_decay: float,
                 patience: int, device: torch.device):
        self.__epochs = epochs
        self.__lr = learning_rate
        self.__wd = weight_decay
        self.__patience = patience
        self.__device = device

    def fit(self, module: nn.Module, train_loader, val_loader,
            weights: torch.Tensor) -> tuple:
        module.to(self.__device)
        params = [p for p in module.parameters() if p.requires_grad]
        optimiser = torch.optim.Adam(params, lr=self.__lr, weight_decay=self.__wd)
        loss_fn = nn.CrossEntropyLoss(weight=weights.to(self.__device))
        best_state, best_f1, waited = None, -1.0, 0
        history = []
        for epoch in range(1, self.__epochs + 1):
            start = time.time()
            train_loss = self.__train_epoch(module, train_loader, optimiser, loss_fn)
            val_loss, logits, labels = self.evaluate(module, val_loader, loss_fn)
            f1 = macro_f1(labels, logits.argmax(axis=1))
            history.append({"epoch": epoch, "train_loss": train_loss,
                            "val_loss": val_loss, "val_macro_f1": f1,
                            "seconds": round(time.time() - start, 1)})
            print(f"  epoch {epoch:3d}  train {train_loss:.4f}  "
                  f"val {val_loss:.4f}  val macro-F1 {f1:.4f}", flush=True)
            if f1 > best_f1:
                best_f1, best_state, waited = f1, copy.deepcopy(module.state_dict()), 0
            else:
                waited += 1
                if waited >= self.__patience:
                    break
        module.load_state_dict(best_state)
        return module, history

    def __train_epoch(self, module, loader, optimiser, loss_fn) -> float:
        module.train()
        total, count = 0.0, 0
        for x, y in loader:
            x, y = x.to(self.__device), y.to(self.__device)
            optimiser.zero_grad()
            loss = loss_fn(module(x), y)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(y)
            count += len(y)
        return total / count

    @torch.no_grad()
    def evaluate(self, module, loader, loss_fn=None) -> tuple:
        module.eval()
        logits, labels, total = [], [], 0.0
        for x, y in loader:
            out = module(x.to(self.__device))
            if loss_fn is not None:
                total += loss_fn(out, y.to(self.__device)).item() * len(y)
            logits.append(out.cpu())
            labels.append(y)
        logits = torch.cat(logits).numpy()
        labels = torch.cat(labels).numpy()
        return total / max(len(labels), 1), logits, labels


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
