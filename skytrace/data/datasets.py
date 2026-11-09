"""PyTorch datasets over a ClipSet.

Authors: Maria Habak, [Partner name]
Reference: PyTorch "Datasets & DataLoaders" tutorial,
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

Pixels are scaled to [0, 1], repeated to 3 channels and normalised with
the ImageNet statistics, because the frame encoder is pretrained on
ImageNet. Training augmentation (flips, small brightness/contrast
changes) is applied identically to every frame of a clip, so that it
cannot create fake motion between frames.
"""

import torch
from torch.utils.data import Dataset

from skytrace.data.clips import ClipSet

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_network_input(clip_uint8: torch.Tensor, size: int) -> torch.Tensor:
    """(T, S, S) uint8 -> (T, 3, size, size) normalised float."""
    x = clip_uint8.float().div(255.0).unsqueeze(1)
    if x.shape[-1] != size:
        x = torch.nn.functional.interpolate(x, size=(size, size),
                                            mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


class ClipAugmenter:
    """Random flips and brightness/contrast jitter, the same for all frames."""

    def __init__(self, brightness: float, contrast: float, generator=None):
        self.__brightness = brightness
        self.__contrast = contrast
        self.__generator = generator

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        g = self.__generator
        x = clip.float()
        if torch.rand(1, generator=g).item() < 0.5:
            x = x.flip(-1)
        if torch.rand(1, generator=g).item() < 0.5:
            x = x.flip(-2)
        scale = 1 + (torch.rand(1, generator=g).item() * 2 - 1) * self.__contrast
        shift = (torch.rand(1, generator=g).item() * 2 - 1) * self.__brightness * 255
        mean = x.mean()
        x = (x - mean) * scale + mean + shift
        return x.round().clamp(0, 255).to(torch.uint8)


class ClipDataset(Dataset):
    """One item per clip: (T, 3, H, W) tensor and its label."""

    def __init__(self, clip_set: ClipSet, input_size: int, augmenter=None,
                 shuffle_frames: bool = False, seed: int = 0):
        self.__clips = clip_set.clips
        self.__labels = clip_set.labels
        self.__size = input_size
        self.__augmenter = augmenter
        self.__shuffle = shuffle_frames
        self.__generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self.__labels)

    def __getitem__(self, i):
        clip = self.__clips[i]
        if self.__augmenter is not None:
            clip = self.__augmenter(clip)
        if self.__shuffle:
            clip = clip[torch.randperm(clip.shape[0], generator=self.__generator)]
        return to_network_input(clip, self.__size), self.__labels[i]


class FrameDataset(Dataset):
    """One item per frame of every clip, for training the frame model."""

    def __init__(self, clip_set: ClipSet, input_size: int, augmenter=None):
        self.__clips = clip_set.clips
        self.__labels = clip_set.labels
        self.__length = self.__clips.shape[1]
        self.__size = input_size
        self.__augmenter = augmenter

    def __len__(self) -> int:
        return len(self.__labels) * self.__length

    def __getitem__(self, i):
        clip_index, frame_index = divmod(i, self.__length)
        frame = self.__clips[clip_index, frame_index: frame_index + 1]
        if self.__augmenter is not None:
            frame = self.__augmenter(frame)
        return to_network_input(frame, self.__size)[0], self.__labels[clip_index]
