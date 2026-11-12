"""TargetClassifier: the interface shared by every model.

Author: Mariah Habak

The experiment code and the Evaluator only call these methods, so they
never need to know whether they run the majority baseline, the small
CNN or ResNet-18.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import torch

from dronebird.data.crops import CropBank


class TargetClassifier(ABC):
    """Classifies object crops as bird (0) or drone (1).

    ``predict_scores`` takes float crops (N, 3, S, S) in [0, 1] and returns
    (N, 2) scores that sum to 1 (model scores, not guaranteed
    probabilities).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def fit(self, train: CropBank, val: CropBank, run_dir: Path) -> list:
        """Train; keep the best state on validation; return the history."""

    @abstractmethod
    def predict_scores(self, images: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def save(self, run_dir: Path) -> None:
        ...

    @abstractmethod
    def load(self, run_dir: Path) -> None:
        ...

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return self.predict_scores(images).argmax(dim=1)
