"""Calibration (temperature scaling) and the "uncertain" decision.

Authors: Maria Habak, [Partner name]
Reference: Guo et al. (2017), "On Calibration of Modern Neural
Networks", ICML (temperature scaling).

Confidence  = the highest class score for a clip.
Calibration = whether those scores match how often the model is right.
Rejection   = answer "uncertain" when confidence < threshold.
Both the temperature and the threshold are fitted on validation data
only, then frozen and applied to the test set.
"""

import numpy as np

UNCERTAIN = -1


def _nll(y, scores):
    return float(-np.log(scores[np.arange(len(y)), y] + 1e-12).mean())


def rescale(scores: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(scores, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


class TemperatureScaler:
    """Finds one temperature T > 0 that minimises validation log-loss."""

    def __init__(self, grid=None):
        self.__grid = np.exp(np.linspace(np.log(0.05), np.log(20), 400)) \
            if grid is None else np.asarray(grid)
        self.__temperature = 1.0

    def fit(self, y_val, scores_val: np.ndarray) -> float:
        y = np.asarray(y_val).astype(int)
        losses = [_nll(y, rescale(scores_val, t)) for t in self.__grid]
        self.__temperature = float(self.__grid[int(np.argmin(losses))])
        return self.__temperature

    @property
    def temperature(self) -> float:
        return self.__temperature

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return rescale(scores, self.__temperature)


class RejectionRule:
    """Threshold chosen so that a target share of validation clips is answered."""

    def __init__(self, target_coverage: float):
        if not 0 < target_coverage <= 1:
            raise ValueError("target_coverage must be in (0, 1]")
        self.__coverage = target_coverage
        self.__threshold = 0.0

    def fit(self, scores_val: np.ndarray) -> float:
        confidence = scores_val.max(axis=1)
        self.__threshold = float(np.quantile(confidence, 1 - self.__coverage))
        return self.__threshold

    @property
    def threshold(self) -> float:
        return self.__threshold

    def decide(self, scores: np.ndarray) -> np.ndarray:
        """Class index per clip, or UNCERTAIN (-1)."""
        decision = scores.argmax(axis=1)
        decision[scores.max(axis=1) < self.__threshold] = UNCERTAIN
        return decision

    def report(self, y_true, scores: np.ndarray) -> dict:
        y = np.asarray(y_true)
        decision = self.decide(scores)
        answered = decision != UNCERTAIN
        return {"threshold": self.__threshold,
                "coverage": float(answered.mean()),
                "error_when_answered": float((decision[answered] != y[answered]).mean())
                if answered.any() else float("nan"),
                "error_without_rejection": float((scores.argmax(axis=1) != y).mean())}
