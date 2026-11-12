"""Training augmentation and test degradations, all written with tensors.

Author: Mariah Habak

Every class takes a float batch (N, 3, h, w) with values in [0, 1] and
returns a float batch. The object size ``s`` is the longest side of the
object, in pixels; on the canvas a full-size object is
canvas / margin = 64 px, so shrinking the object to ``s`` means
shrinking the whole crop to round(margin * s) pixels.

TRAINING recipe (DistanceAugment), with probability p:
  1. shrink with F.interpolate(mode="area") to an object of s ~ U[12, 64] px
  2. Gaussian blur, sigma ~ U[0, 1] px, as a depthwise convolution
  3. Gaussian noise, std ~ U[0, 0.04], with torch.randn_like
  4. bilinear back to the canvas (the shared ``to_canvas`` step)

TEST recipe (synthetic small targets), deliberately different:
  1. shrink with bicubic interpolation (torchvision resize, antialias)
  2. JPEG compression at quality 50 (torchvision.io)
  3. optional Poisson "shot" noise (off by default)
  4. bilinear back to the canvas (the same shared step)
No Gaussian blur, no Gaussian noise and no area resampling at test time,
so the augmented model is not tested on what it was trained on.
"""

import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torchvision.io import decode_jpeg, encode_jpeg
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from dronebird.data.images import to_canvas


class Augmentation(ABC):
    """A change applied to a float image batch (N, 3, h, w) in [0, 1]."""

    @abstractmethod
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        ...

    @property
    def name(self) -> str:
        return type(self).__name__


def small_side(object_size: float, margin: float) -> int:
    """Crop side (px) at which the object's longest side is ``object_size``."""
    return max(2, int(round(margin * object_size)))


def gaussian_kernel(sigma: float, channels: int) -> torch.Tensor:
    """Depthwise 2-D Gaussian kernel (channels, 1, k, k)."""
    radius = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-radius, radius + 1, dtype=torch.float32)
    g = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]  # outer product -> (k, k)
    return kernel.expand(channels, 1, -1, -1).contiguous()


def gaussian_blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.05:
        return images
    kernel = gaussian_kernel(sigma, images.shape[1])
    radius = kernel.shape[-1] // 2
    mode = "reflect" if min(images.shape[-2:]) > radius else "replicate"
    padded = F.pad(images, (radius,) * 4, mode=mode)
    return F.conv2d(padded, kernel, groups=images.shape[1])


class RandomFlip(Augmentation):
    """Horizontal flip with probability p (used by every network)."""

    def __init__(self, p: float = 0.5):
        self.__p = p

    def __call__(self, images):
        if torch.rand(1).item() < self.__p:
            return images.flip(-1)
        return images


class DistanceAugment(Augmentation):
    """Makes a crop look as if the object were further away."""

    def __init__(self, p: float, min_size: float, max_size: float, margin: float,
                 canvas: int, blur_sigma_max: float, noise_std_max: float):
        if not 0 <= p <= 1 or min_size <= 0 or max_size < min_size:
            raise ValueError("Invalid DistanceAugment settings")
        self.__p = p
        self.__min = min_size
        self.__max = max_size
        self.__margin = margin
        self.__canvas = canvas
        self.__blur = blur_sigma_max
        self.__noise = noise_std_max

    def __call__(self, images):
        if torch.rand(1).item() >= self.__p:
            return images
        u = torch.rand(3).tolist()
        size = self.__min + u[0] * (self.__max - self.__min)
        side = small_side(size, self.__margin)
        x = F.interpolate(images, size=(side, side), mode="area")
        x = gaussian_blur(x, u[1] * self.__blur)
        x = x + torch.randn_like(x) * (u[2] * self.__noise)
        x = x.clamp(0.0, 1.0)
        return to_canvas(x, self.__canvas).clamp(0.0, 1.0)


class BicubicShrink(Augmentation):
    """Test degradation 1: bicubic shrink to an object of ``object_size`` px."""

    def __init__(self, object_size: float, margin: float):
        self.__side = small_side(object_size, margin)

    def __call__(self, images):
        out = resize(images, [self.__side, self.__side],
                     interpolation=InterpolationMode.BICUBIC, antialias=True)
        return out.clamp(0.0, 1.0)


class JpegCompression(Augmentation):
    """Test degradation 2: JPEG encode and decode at a fixed quality."""

    def __init__(self, quality: int):
        if not 1 <= quality <= 100:
            raise ValueError("JPEG quality must be in 1..100")
        self.__quality = quality

    def __call__(self, images):
        as_bytes = (images * 255).round().clamp(0, 255).to(torch.uint8)
        out = [decode_jpeg(encode_jpeg(img, quality=self.__quality))
               for img in as_bytes]
        return torch.stack(out).float() / 255.0


class ShotNoise(Augmentation):
    """Test degradation 3: Poisson noise with a fixed seed (same for all models)."""

    def __init__(self, photons: float, seed: int):
        self.__photons = photons
        self.__seed = seed

    def __call__(self, images):
        if self.__photons <= 0:
            return images
        g = torch.Generator().manual_seed(self.__seed)
        noisy = torch.poisson(images * self.__photons, generator=g) / self.__photons
        return noisy.clamp(0.0, 1.0)


class CanvasResize(Augmentation):
    """The shared last step: bilinear resize to the network's input size."""

    def __init__(self, canvas: int):
        self.__canvas = canvas

    def __call__(self, images):
        return to_canvas(images, self.__canvas).clamp(0.0, 1.0)


class Compose(Augmentation):
    def __init__(self, steps: list):
        self.__steps = list(steps)

    def __call__(self, images):
        for step in self.__steps:
            images = step(images)
        return images

    @property
    def name(self) -> str:
        return " -> ".join(s.name for s in self.__steps)


def synthetic_test_degradation(object_size: float, settings: dict, margin: float,
                               canvas: int) -> Compose:
    """The test recipe for one target size (see the module docstring)."""
    steps = [BicubicShrink(object_size, margin)]
    if settings.get("jpeg_quality"):
        steps.append(JpegCompression(int(settings["jpeg_quality"])))
    if settings.get("shot_noise_photons"):
        steps.append(ShotNoise(float(settings["shot_noise_photons"]),
                               int(settings.get("noise_seed", 0))))
    steps.append(CanvasResize(canvas))
    return Compose(steps)


def build_distance_augment(settings: dict, margin: float, canvas: int) -> DistanceAugment:
    return DistanceAugment(p=settings["p"], min_size=settings["min_size"],
                           max_size=settings["max_size"], margin=margin,
                           canvas=canvas, blur_sigma_max=settings["blur_sigma_max"],
                           noise_std_max=settings["noise_std_max"])
