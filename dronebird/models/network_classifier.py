"""A TargetClassifier around one of our networks.

Author: Mariah Habak

The same class runs baseline 2 (SmallCNN), ResNet-18 without
augmentation and ResNet-18 with DistanceAugment; only the JSON
configuration changes. Every network gets a random horizontal flip, so
the only difference between the two ResNet-18 versions is
DistanceAugment.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dronebird.augment import RandomFlip, build_distance_augment
from dronebird.data.crops import CropBank
from dronebird.data.datasets import FlyingObjectCrops, normalise
from dronebird.models.base import TargetClassifier
from dronebird.models.networks import build_network
from dronebird.training import Trainer, class_weights, pick_device

CHECKPOINT = "best.pt"


class NetworkClassifier(TargetClassifier):
    def __init__(self, network: str, margin: float, canvas: int, seed: int,
                 distance_augment: dict = None, pretrained: bool = True,
                 dropout: float = 0.3, epochs: int = 12, batch_size: int = 128,
                 learning_rate: float = 3e-4, weight_decay: float = 1e-4,
                 patience: int = 4, num_workers: int = 2, device: str = "auto"):
        self.__kind = network
        self.__seed = seed
        self.__batch = batch_size
        self.__workers = num_workers
        self.__device = pick_device(device)
        self.__net = build_network(network, pretrained=pretrained, dropout=dropout)
        self.__augmentations = [RandomFlip(0.5)]
        self.__distance = distance_augment is not None
        if self.__distance:
            self.__augmentations.append(
                build_distance_augment(distance_augment, margin, canvas))
        self.__trainer = Trainer(epochs, learning_rate, weight_decay, patience,
                                 self.__device)

    @property
    def name(self) -> str:
        return self.__kind + ("_distance" if self.__distance else "")

    @property
    def network(self) -> torch.nn.Module:
        return self.__net

    def fit(self, train: CropBank, val: CropBank, run_dir: Path) -> list:
        g = torch.Generator().manual_seed(self.__seed)
        train_loader = DataLoader(FlyingObjectCrops(train, self.__augmentations),
                                  batch_size=self.__batch, shuffle=True,
                                  num_workers=self.__workers, generator=g,
                                  pin_memory=self.__device.type == "cuda")
        val_loader = DataLoader(FlyingObjectCrops(val), batch_size=self.__batch,
                                num_workers=self.__workers)
        return self.__trainer.fit(self.__net, train_loader, val_loader,
                                  class_weights(train.labels),
                                  Path(run_dir) / CHECKPOINT)

    @torch.no_grad()
    def predict_scores(self, images: torch.Tensor) -> torch.Tensor:
        self.__net.to(self.__device).eval()
        out = []
        for i in range(0, len(images), self.__batch):
            x = normalise(images[i: i + self.__batch]).to(self.__device)
            out.append(self.__net(x).softmax(dim=1).cpu())
        return torch.cat(out) if out else torch.zeros(0, 2)

    def save(self, run_dir: Path) -> None:
        path = Path(run_dir) / CHECKPOINT
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.__net.state_dict(), path)

    def load(self, run_dir: Path) -> None:
        state = torch.load(Path(run_dir) / CHECKPOINT, map_location="cpu",
                           weights_only=True)
        self.__net.load_state_dict(state)
