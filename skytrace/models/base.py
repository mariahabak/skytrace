"""The interface shared by every SkyTrace model.

Authors: Maria Habak, [Partner name]

The experiment code only calls these methods. It never asks which
concrete class it holds, so a new model is a new class and a new JSON
file, not a change to the experiment code.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from skytrace.data.clips import ClipSet


class ClipClassifier(ABC):
    """Classifies clips as bird (0) or drone (1).

    ``predict_scores`` returns one row of two scores per clip that sum to
    1. They are model scores, not guaranteed probabilities; calibration
    is measured separately.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def fit(self, train: ClipSet, val: ClipSet, run_dir: Path) -> list:
        """Train, keep the best state on validation, return the history."""

    @abstractmethod
    def predict_scores(self, clips: ClipSet) -> np.ndarray:
        """Array (N, 2) of class scores."""

    @abstractmethod
    def save(self, run_dir: Path) -> None:
        ...

    @abstractmethod
    def load(self, run_dir: Path) -> None:
        ...

    def predict(self, clips: ClipSet) -> np.ndarray:
        return self.predict_scores(clips).argmax(axis=1)
