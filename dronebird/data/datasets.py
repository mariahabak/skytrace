"""FlyingObjectCrops: the PyTorch dataset over a CropBank.

Author: Mariah Habak
Reference: PyTorch "Datasets & DataLoaders" tutorial,
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

Every model receives its input the same way:
uint8 crop (3, 96, 96) -> float in [0, 1] -> optional augmentation
-> normalisation with the ImageNet mean and std, by broadcasting:
    (x - mean) / std    with x (3, 96, 96) and mean, std of shape (3, 1, 1)
"""

import torch
from torch.utils.data import Dataset

from dronebird.data.crops import CropBank

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def normalise(images: torch.Tensor) -> torch.Tensor:
    """Float images (..., 3, H, W) in [0, 1] -> normalised, by broadcasting."""
    return (images - IMAGENET_MEAN) / IMAGENET_STD


def to_unit(images_uint8: torch.Tensor) -> torch.Tensor:
    return images_uint8.float() / 255.0


class FlyingObjectCrops(Dataset):
    """One item per object crop: (normalised image (3, S, S), label)."""

    def __init__(self, bank: CropBank, augmentations=()):
        self.__images = bank.images
        self.__labels = bank.labels
        self.__augmentations = list(augmentations)

    def __len__(self) -> int:
        return len(self.__labels)

    def __getitem__(self, i):
        x = to_unit(self.__images[i]).unsqueeze(0)  # (1, 3, S, S)
        for augmentation in self.__augmentations:
            x = augmentation(x)
        return normalise(x[0]), self.__labels[i]

    @property
    def labels(self) -> torch.Tensor:
        return self.__labels.clone()
