"""Non-neural baseline: hand-made clip features + logistic regression.

Authors: Maria Habak, [Partner name]

Everything is NumPy. The features describe appearance (how bright and
how textured the object is compared with the sky around it), shape
change over time (wing flapping changes the box and the pixels), and
movement of the box in the frame. The optional geometry features (box
size and aspect ratio) let us measure how much a model could get from
size alone, which is a known shortcut risk in this dataset.
"""

import json
from pathlib import Path

import numpy as np

from skytrace.data.clips import ClipSet
from skytrace.models.base import ClipClassifier


class ClipFeatures:
    """Turns a ClipSet into a feature matrix (N, F)."""

    APPEARANCE = ["object_contrast", "object_texture", "surround_texture",
                  "pixel_change", "texture_change"]
    MOTION = ["speed", "turning", "size_change", "aspect_change"]
    GEOMETRY = ["log_box_side", "log_aspect"]

    def __init__(self, use_geometry: bool, centre_fraction: float):
        self.__use_geometry = use_geometry
        self.__centre = centre_fraction

    @property
    def names(self) -> list:
        names = self.APPEARANCE + self.MOTION
        return names + self.GEOMETRY if self.__use_geometry else names

    def transform(self, clip_set: ClipSet) -> np.ndarray:
        clips = clip_set.clips.numpy().astype(np.float32) / 255.0
        tracks = clip_set.tracks.numpy()
        size = clips.shape[-1]
        half = int(round(size * self.__centre / 2))
        lo, hi = size // 2 - half, size // 2 + half
        centre = clips[:, :, lo:hi, lo:hi]
        mask = np.ones((size, size), dtype=bool)
        mask[lo:hi, lo:hi] = False
        surround = clips[:, :, mask]

        object_contrast = (centre.mean(axis=(2, 3)) - surround.mean(axis=2)).mean(axis=1)
        object_texture = centre.std(axis=(2, 3)).mean(axis=1)
        surround_texture = surround.std(axis=2).mean(axis=1)
        pixel_change = np.abs(np.diff(centre, axis=1)).mean(axis=(1, 2, 3))
        texture_change = centre.std(axis=(2, 3)).std(axis=1)

        xy = tracks[:, :, :2]
        steps = np.diff(xy, axis=1)
        speed = np.linalg.norm(steps, axis=2).mean(axis=1)
        angles = np.arctan2(steps[:, :, 1], steps[:, :, 0])
        turn = np.angle(np.exp(1j * np.diff(angles, axis=1)))
        turning = np.abs(turn).mean(axis=1)
        wh = tracks[:, :, 2:]
        area = wh[:, :, 0] * wh[:, :, 1]
        size_change = area.std(axis=1) / (area.mean(axis=1) + 1e-8)
        aspect = np.log((wh[:, :, 0] + 1e-8) / (wh[:, :, 1] + 1e-8))
        aspect_change = aspect.std(axis=1)

        columns = [object_contrast, object_texture, surround_texture, pixel_change,
                   texture_change, speed, turning, size_change, aspect_change]
        if self.__use_geometry:
            columns += [np.log(clip_set.box_side.numpy() + 1.0), aspect.mean(axis=1)]
        return np.stack(columns, axis=1).astype(np.float64)


class LogisticRegression:
    """Binary logistic regression trained with full-batch gradient descent."""

    def __init__(self, learning_rate: float, epochs: int, l2: float,
                 balance_classes: bool):
        self.__lr = learning_rate
        self.__epochs = epochs
        self.__l2 = l2
        self.__balance = balance_classes
        self.__weights = None
        self.__bias = 0.0
        self.__mean = None
        self.__std = None

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, x: np.ndarray, y: np.ndarray) -> list:
        self.__mean = x.mean(axis=0)
        self.__std = x.std(axis=0) + 1e-8
        z = (x - self.__mean) / self.__std
        n, f = z.shape
        if self.__balance:
            frac = y.mean()
            sample_w = np.where(y == 1, 0.5 / frac, 0.5 / (1 - frac))
        else:
            sample_w = np.ones(n)
        self.__weights = np.zeros(f)
        self.__bias = 0.0
        losses = []
        for _ in range(self.__epochs):
            p = self._sigmoid(z @ self.__weights + self.__bias)
            err = (p - y) * sample_w
            self.__weights -= self.__lr * (z.T @ err / n + self.__l2 * self.__weights)
            self.__bias -= self.__lr * err.mean()
            eps = 1e-12
            losses.append(float(-(sample_w * (y * np.log(p + eps)
                                              + (1 - y) * np.log(1 - p + eps))).mean()))
        return losses

    def predict_drone_score(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.__mean) / self.__std
        return self._sigmoid(z @ self.__weights + self.__bias)

    @property
    def weights(self) -> np.ndarray:
        return self.__weights.copy()

    def state(self) -> dict:
        return {"weights": self.__weights.tolist(), "bias": self.__bias,
                "mean": self.__mean.tolist(), "std": self.__std.tolist()}

    def set_state(self, state: dict) -> None:
        self.__weights = np.asarray(state["weights"])
        self.__bias = float(state["bias"])
        self.__mean = np.asarray(state["mean"])
        self.__std = np.asarray(state["std"])


class FeatureBaseline(ClipClassifier):
    """Hand-made clip features followed by logistic regression."""

    def __init__(self, use_geometry: bool = True, centre_fraction: float = 0.5,
                 learning_rate: float = 0.1, epochs: int = 2000, l2: float = 1e-3,
                 balance_classes: bool = True):
        self.__features = ClipFeatures(use_geometry, centre_fraction)
        self.__model = LogisticRegression(learning_rate, epochs, l2, balance_classes)
        self.__use_geometry = use_geometry

    @property
    def name(self) -> str:
        return "feature_baseline" + ("" if self.__use_geometry else "_no_geometry")

    def fit(self, train: ClipSet, val: ClipSet, run_dir: Path) -> list:
        losses = self.__model.fit(self.__features.transform(train),
                                  train.labels.numpy().astype(np.float64))
        return [{"epoch": i + 1, "train_loss": loss}
                for i, loss in enumerate(losses) if (i + 1) % 100 == 0]

    def predict_scores(self, clips: ClipSet) -> np.ndarray:
        drone = self.__model.predict_drone_score(self.__features.transform(clips))
        return np.stack([1 - drone, drone], axis=1)

    def feature_weights(self) -> dict:
        return dict(zip(self.__features.names, self.__model.weights.tolist()))

    def save(self, run_dir: Path) -> None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "baseline.json", "w", encoding="utf-8") as h:
            json.dump({"state": self.__model.state(),
                       "feature_weights": self.feature_weights()}, h, indent=2)

    def load(self, run_dir: Path) -> None:
        with open(Path(run_dir) / "baseline.json", encoding="utf-8") as h:
            self.__model.set_state(json.load(h)["state"])
