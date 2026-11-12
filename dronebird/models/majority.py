"""Baseline 1: always predict the most common training class.

Author: Mariah Habak
"""

import json
from pathlib import Path

import torch

from dronebird.models.base import TargetClassifier


class MajorityClassifier(TargetClassifier):
    def __init__(self):
        self.__label = 0

    @property
    def name(self) -> str:
        return "majority"

    def fit(self, train, val, run_dir):
        self.__label = int(torch.bincount(train.labels, minlength=2).argmax().item())
        return []

    def predict_scores(self, images):
        scores = torch.zeros(len(images), 2)
        scores[:, self.__label] = 1.0
        return scores

    def save(self, run_dir):
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "majority.json").write_text(json.dumps({"label": self.__label}))

    def load(self, run_dir):
        self.__label = json.loads((Path(run_dir) / "majority.json").read_text())["label"]
