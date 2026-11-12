"""Evaluator: accuracy and F1 at every target size.

Author: Mariah Habak

Two test sets, reported separately:
- synthetic: objects that are large in the original frame
  (>= ``synthetic_min_size`` px), shrunk with the TEST recipe to
  48, 32, 24 and 16 px ("full" = unchanged);
- real small: objects that are naturally small in the original frame
  (< ``real_max_size`` px), not changed at all, grouped into size bins.

Results are dictionaries keyed by (test set, size):
    {("synthetic", "full"): {"accuracy": .., "macro_f1": .., "n": ..}, ...}
"""

import torch

from dronebird import metrics
from dronebird.augment import synthetic_test_degradation
from dronebird.data.crops import CropBank
from dronebird.data.datasets import to_unit
from dronebird.models.base import TargetClassifier

FULL = "full"


class Evaluator:
    def __init__(self, evaluation: dict, margin: float, canvas: int):
        self.__sizes = [int(s) for s in evaluation["synthetic_sizes"]]
        self.__min_size = float(evaluation["synthetic_min_size"])
        self.__real_bins = {str(k): (float(v[0]), float(v[1]))
                            for k, v in evaluation["real_size_bins"].items()}
        self.__real_max = float(evaluation["real_max_size"])
        self.__degradation = dict(evaluation["test_degradation"])
        self.__margin = margin
        self.__canvas = canvas
        self.__cache = {}

    @property
    def size_columns(self) -> list:
        return [FULL] + [str(s) for s in self.__sizes]

    @property
    def real_columns(self) -> list:
        return list(self.__real_bins) + ["all"]

    def test_sets(self, bank: CropBank) -> dict:
        size = bank.object_size
        return {"all": bank,
                "synthetic": bank.select(size >= self.__min_size),
                "real_small": bank.select(size < self.__real_max)}

    def __degraded(self, key: str, bank: CropBank, size: int) -> torch.Tensor:
        """Degraded images, computed once and reused by every model."""
        cache_key = (key, size, len(bank))
        if cache_key not in self.__cache:
            recipe = synthetic_test_degradation(size, self.__degradation,
                                                self.__margin, self.__canvas)
            images = to_unit(bank.images)
            out = [recipe(images[i: i + 256]) for i in range(0, len(images), 256)]
            self.__cache[cache_key] = torch.cat(out) if out else images
        return self.__cache[cache_key]

    @staticmethod
    def __score(model: TargetClassifier, images, labels) -> dict:
        if len(labels) == 0:
            return {"accuracy": float("nan"), "macro_f1": float("nan"), "n": 0}
        pred = model.predict(images)
        return {"accuracy": metrics.accuracy(labels, pred),
                "macro_f1": metrics.macro_f1(labels, pred), "n": int(len(labels))}

    def size_curve(self, model: TargetClassifier, bank: CropBank, split: str) -> dict:
        sets = self.test_sets(bank)
        results = {}
        everything = sets["all"]
        results[("all", FULL)] = self.__score(model, to_unit(everything.images),
                                              everything.labels)
        synthetic = sets["synthetic"]
        results[("synthetic", FULL)] = self.__score(model, to_unit(synthetic.images),
                                                    synthetic.labels)
        for size in self.__sizes:
            images = self.__degraded(f"{split}/synthetic", synthetic, size)
            results[("synthetic", str(size))] = self.__score(model, images,
                                                             synthetic.labels)
        small = sets["real_small"]
        for name, (lo, hi) in self.__real_bins.items():
            part = small.select((small.object_size >= lo) & (small.object_size < hi))
            results[("real_small", name)] = self.__score(model, to_unit(part.images),
                                                         part.labels)
        results[("real_small", "all")] = self.__score(model, to_unit(small.images),
                                                      small.labels)
        return results

    def wrong_predictions(self, model: TargetClassifier, bank: CropBank,
                          test_set: str, size: str) -> dict:
        """Indices, images, labels and scores of the wrong predictions."""
        sets = self.test_sets(bank)
        chosen = sets[test_set]
        if test_set == "synthetic" and size != FULL:
            recipe = synthetic_test_degradation(int(size), self.__degradation,
                                                self.__margin, self.__canvas)
            images = recipe(to_unit(chosen.images))
        elif test_set == "real_small" and size != "all":
            lo, hi = self.__real_bins[size]
            chosen = chosen.select((chosen.object_size >= lo) & (chosen.object_size < hi))
            images = to_unit(chosen.images)
        else:
            images = to_unit(chosen.images)
        scores = model.predict_scores(images)
        wrong = torch.nonzero(scores.argmax(1) != chosen.labels, as_tuple=True)[0]
        return {"bank": chosen, "images": images, "scores": scores, "wrong": wrong}


def to_records(results: dict, model: str, seed: int, split: str) -> list:
    return [{"model": model, "seed": seed, "split": split, "test_set": test_set,
             "size": size, **values} for (test_set, size), values in results.items()]
