"""Pretrained ResNet-18 fine-tuned on single frames.

Authors: Maria Habak, [Partner name]

Pretrained model (declared): torchvision ResNet-18 with ImageNet
weights, https://pytorch.org/vision/stable/models/resnet.html
Our parts: the infrared clip pipeline, a new 2-class head with dropout,
and the clip decision rule.

One trained network gives two of our compared models:
- clip_rule "middle": classify only the middle frame of the clip
  (single-frame model);
- clip_rule "mean": average the frame scores over the clip
  (frame-averaging model).
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from skytrace.data.clips import ClipSet
from skytrace.data.datasets import ClipAugmenter, FrameDataset, to_network_input
from skytrace.models.base import ClipClassifier
from skytrace.training import Trainer, class_weights, pick_device, softmax

CLIP_RULES = ("middle", "mean")


class FrameNet(nn.Module):
    """ResNet-18 backbone and a new head: frame (3, H, W) -> 2 logits."""

    FEATURE_SIZE = 512

    def __init__(self, pretrained: bool, dropout: float):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(self.FEATURE_SIZE, 2))

    def forward(self, x):
        return self.head(self.encoder(x))


class FrameCNN(ClipClassifier):
    def __init__(self, clip_rule: str = "mean", pretrained: bool = True,
                 input_size: int = 112, dropout: float = 0.3, epochs: int = 15,
                 batch_size: int = 128, learning_rate: float = 3e-4,
                 weight_decay: float = 1e-4, patience: int = 4,
                 brightness: float = 0.1, contrast: float = 0.2,
                 num_workers: int = 2, device: str = "auto",
                 checkpoint_from: str = "", seed: int = 0):
        if clip_rule not in CLIP_RULES:
            raise ValueError(f"clip_rule must be one of {CLIP_RULES}")
        self.__rule = clip_rule
        self.__size = input_size
        self.__batch = batch_size
        self.__workers = num_workers
        self.__brightness = brightness
        self.__contrast = contrast
        self.__seed = seed
        self.__checkpoint_from = checkpoint_from
        self.__device = pick_device(device)
        self.__net = FrameNet(pretrained, dropout)
        self.__trainer = Trainer(epochs, learning_rate, weight_decay, patience,
                                 self.__device)

    @property
    def name(self) -> str:
        return f"frame_cnn_{self.__rule}"

    @property
    def network(self) -> FrameNet:
        return self.__net

    def fit(self, train: ClipSet, val: ClipSet, run_dir: Path) -> list:
        if self.__checkpoint_from:
            # Same network, different clip rule: reuse the trained weights.
            self.load(Path(self.__checkpoint_from.format(seed=self.__seed)))
            return []
        g = torch.Generator().manual_seed(self.__seed)
        augmenter = ClipAugmenter(self.__brightness, self.__contrast, generator=g)
        train_loader = DataLoader(FrameDataset(train, self.__size, augmenter),
                                  batch_size=self.__batch, shuffle=True,
                                  num_workers=self.__workers, generator=g)
        val_loader = DataLoader(FrameDataset(val, self.__size),
                                batch_size=self.__batch, num_workers=self.__workers)
        self.__net, history = self.__trainer.fit(
            self.__net, train_loader, val_loader, class_weights(train.labels))
        return history

    @torch.no_grad()
    def frame_scores(self, clips: ClipSet) -> np.ndarray:
        """Scores (N, T, 2) for every frame of every clip."""
        logits = self.__run(clips, lambda x: self.__net(x))
        n, t = len(clips), clips.clips.shape[1]
        return softmax(logits.reshape(n * t, 2)).reshape(n, t, 2)

    @torch.no_grad()
    def encode(self, clips: ClipSet) -> np.ndarray:
        """Frame features (N, T, 512) from the encoder, for the temporal model."""
        return self.__run(clips, lambda x: self.__net.encoder(x))

    def __run(self, clips: ClipSet, function) -> np.ndarray:
        self.__net.to(self.__device).eval()
        pixels = clips.clips
        n, t = pixels.shape[:2]
        outputs = []
        per_batch = max(1, self.__batch // t)
        for i in range(0, n, per_batch):
            batch = torch.cat([to_network_input(c, self.__size)
                               for c in pixels[i: i + per_batch]])
            outputs.append(function(batch.to(self.__device)).cpu())
        return torch.cat(outputs).reshape(n, t, -1).numpy()

    def predict_scores(self, clips: ClipSet) -> np.ndarray:
        scores = self.frame_scores(clips)
        if self.__rule == "middle":
            return scores[:, scores.shape[1] // 2]
        return scores.mean(axis=1)

    def save(self, run_dir: Path) -> None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.__net.state_dict(), run_dir / "frame_cnn.pt")

    def load(self, run_dir: Path) -> None:
        state = torch.load(Path(run_dir) / "frame_cnn.pt", map_location="cpu",
                           weights_only=True)
        self.__net.load_state_dict(state)
