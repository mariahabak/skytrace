"""Shared frozen frame encoder followed by a small learned aggregator.

Authors: Maria Habak, [Partner name]

The encoder is the trained frame CNN of the same seed, frozen. Its frame
features are computed once, then only the aggregator is trained. This
makes the ablation clean: every aggregator sees exactly the same
features, so any difference comes from how frames are combined.

Aggregators:
- "mean": average the frame features, then a linear layer. Learned
  averaging: same budget as the others, but blind to frame order.
- "gru":  a one-layer GRU over the frames; its last state -> linear.
- "conv": two 1-D convolutions over time, then average -> linear.

``shuffle_frames`` randomly reorders frames in every clip, in training
and evaluation. If an order-aware aggregator loses nothing when frames
are shuffled, it was not using motion.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from skytrace.data.clips import ClipSet
from skytrace.models.base import ClipClassifier
from skytrace.models.frame_cnn import FrameCNN, FrameNet
from skytrace.training import Trainer, class_weights, pick_device, softmax

AGGREGATORS = ("mean", "gru", "conv")


class Aggregator(nn.Module):
    """(N, T, F) frame features -> (N, 2) logits."""

    def __init__(self, kind: str, features: int, hidden: int, dropout: float):
        super().__init__()
        if kind not in AGGREGATORS:
            raise ValueError(f"aggregator must be one of {AGGREGATORS}")
        self.kind = kind
        self.norm = nn.LayerNorm(features)
        self.drop = nn.Dropout(dropout)
        if kind == "gru":
            self.gru = nn.GRU(features, hidden, batch_first=True)
            out = hidden
        elif kind == "conv":
            self.conv = nn.Sequential(
                nn.Conv1d(features, hidden, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.ReLU())
            out = hidden
        else:
            out = features
        self.head = nn.Linear(out, 2)

    def forward(self, x):
        x = self.drop(self.norm(x))
        if self.kind == "gru":
            _, last = self.gru(x)
            z = last[-1]
        elif self.kind == "conv":
            z = self.conv(x.transpose(1, 2)).mean(dim=2)
        else:
            z = x.mean(dim=1)
        return self.head(self.drop(z))


class TemporalClassifier(ClipClassifier):
    def __init__(self, encoder_checkpoint: str, aggregator: str = "gru",
                 hidden: int = 128, dropout: float = 0.3,
                 shuffle_frames: bool = False, input_size: int = 112,
                 epochs: int = 60, batch_size: int = 64,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                 patience: int = 10, device: str = "auto", seed: int = 0):
        self.__checkpoint = encoder_checkpoint
        self.__kind = aggregator
        self.__shuffle = shuffle_frames
        self.__batch = batch_size
        self.__seed = seed
        self.__device = pick_device(device)
        self.__encoder = FrameCNN(pretrained=False, input_size=input_size,
                                  device=device, seed=seed)
        self.__aggregator = Aggregator(aggregator, FrameNet.FEATURE_SIZE,
                                       hidden, dropout)
        self.__trainer = Trainer(epochs, learning_rate, weight_decay, patience,
                                 self.__device)
        self.__encoder_ready = False
        self.__rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return f"temporal_{self.__kind}" + ("_shuffled" if self.__shuffle else "")

    def __ensure_encoder(self) -> None:
        if not self.__encoder_ready:
            self.__encoder.load(Path(self.__checkpoint.format(seed=self.__seed)))
            self.__encoder_ready = True

    def __features(self, clips: ClipSet) -> torch.Tensor:
        self.__ensure_encoder()
        feats = self.__encoder.encode(clips)
        if self.__shuffle:
            for i in range(len(feats)):
                feats[i] = feats[i][self.__rng.permutation(feats.shape[1])]
        return torch.from_numpy(feats).float()

    def fit(self, train: ClipSet, val: ClipSet, run_dir: Path) -> list:
        g = torch.Generator().manual_seed(self.__seed)
        train_loader = DataLoader(TensorDataset(self.__features(train), train.labels),
                                  batch_size=self.__batch, shuffle=True, generator=g)
        val_loader = DataLoader(TensorDataset(self.__features(val), val.labels),
                                batch_size=self.__batch)
        self.__aggregator, history = self.__trainer.fit(
            self.__aggregator, train_loader, val_loader, class_weights(train.labels))
        return history

    def predict_scores(self, clips: ClipSet) -> np.ndarray:
        loader = DataLoader(TensorDataset(self.__features(clips), clips.labels),
                            batch_size=self.__batch)
        _, logits, _ = self.__trainer.evaluate(self.__aggregator, loader)
        return softmax(logits)

    def save(self, run_dir: Path) -> None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.__aggregator.state_dict(), run_dir / "aggregator.pt")

    def load(self, run_dir: Path) -> None:
        state = torch.load(Path(run_dir) / "aggregator.pt", map_location="cpu",
                           weights_only=True)
        self.__aggregator.load_state_dict(state)
