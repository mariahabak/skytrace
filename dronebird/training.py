"""Trainer: our own training loop.

Author: Mariah Habak
Reference: PyTorch "Optimizing Model Parameters" tutorial,
https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

Every epoch: train (zero_grad, forward, loss, backward, step), then
validate. The weights with the best validation accuracy are saved with
torch.save and restored at the end. Training stops early when the
validation accuracy has not improved for ``patience`` epochs.
"""

import time
from pathlib import Path

import torch
from torch import nn

from dronebird import metrics


def pick_device(preferred: str = "auto") -> torch.device:
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def class_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=2).float()
    return counts.sum() / (2.0 * counts.clamp(min=1))


class Trainer:
    def __init__(self, epochs: int, learning_rate: float, weight_decay: float,
                 patience: int, device: torch.device):
        self.__epochs = epochs
        self.__lr = learning_rate
        self.__wd = weight_decay
        self.__patience = patience
        self.__device = device

    def fit(self, net: nn.Module, train_loader, val_loader, weights: torch.Tensor,
            checkpoint: Path) -> list:
        net.to(self.__device)
        optimiser = torch.optim.Adam(net.parameters(), lr=self.__lr,
                                     weight_decay=self.__wd)
        loss_fn = nn.CrossEntropyLoss(weight=weights.to(self.__device))
        best, waited, history = -1.0, 0, []
        checkpoint = Path(checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, self.__epochs + 1):
            start = time.time()
            train_loss = self.__train_epoch(net, train_loader, optimiser, loss_fn)
            val_loss, scores, labels = self.evaluate(net, val_loader, loss_fn)
            pred = scores.argmax(dim=1)
            acc = metrics.accuracy(labels, pred)
            f1 = metrics.macro_f1(labels, pred)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                            "val_accuracy": acc, "val_macro_f1": f1,
                            "seconds": round(time.time() - start, 1)})
            print(f"  epoch {epoch:3d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}"
                  f"  val acc {acc:.4f}  val F1 {f1:.4f}", flush=True)
            if acc > best:
                best, waited = acc, 0
                torch.save(net.state_dict(), checkpoint)
            else:
                waited += 1
                if waited >= self.__patience:
                    break
        net.load_state_dict(torch.load(checkpoint, map_location=self.__device,
                                       weights_only=True))
        return history

    def __train_epoch(self, net, loader, optimiser, loss_fn) -> float:
        net.train()
        total, count = 0.0, 0
        for x, y in loader:
            x, y = x.to(self.__device), y.to(self.__device)
            optimiser.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(y)
            count += len(y)
        return total / max(count, 1)

    @torch.no_grad()
    def evaluate(self, net, loader, loss_fn=None) -> tuple:
        net.to(self.__device).eval()
        scores, labels, total = [], [], 0.0
        for x, y in loader:
            out = net(x.to(self.__device))
            if loss_fn is not None:
                total += loss_fn(out, y.to(self.__device)).item() * len(y)
            scores.append(out.softmax(dim=1).cpu())
            labels.append(y)
        labels = torch.cat(labels)
        return total / max(len(labels), 1), torch.cat(scores), labels
