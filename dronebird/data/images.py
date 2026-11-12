"""Reading images as tensors and resizing them to the network canvas.

Author: Mariah Habak

``to_canvas`` is the one resize step that every crop goes through before
it reaches a network: the full-size crop, a shrunk test crop, an
augmented training crop and an image uploaded to the demo.
"""

import torch
import torch.nn.functional as F
from torchvision.io import ImageReadMode, read_image


def load_rgb(path) -> torch.Tensor:
    """uint8 tensor (3, H, W)."""
    return read_image(str(path), mode=ImageReadMode.RGB)


def to_canvas(images: torch.Tensor, size: int) -> torch.Tensor:
    """Bilinear resize of float images (..., 3, h, w) to (..., 3, size, size).

    Antialiasing is on, so a large crop is smoothed when it is shrunk;
    a small crop is simply enlarged with bilinear interpolation.
    """
    single = images.dim() == 3
    x = images.unsqueeze(0) if single else images
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear",
                          align_corners=False, antialias=True)
    return x[0] if single else x


def colour_thumbnail(image: torch.Tensor, size: int) -> torch.Tensor:
    """Float colour thumbnail (3, size, size) in [0, 1] of a uint8 image."""
    x = image.float().unsqueeze(0) / 255.0
    return F.interpolate(x, size=(size, size), mode="area")[0]


def gray_thumbnail(image: torch.Tensor, size: int) -> torch.Tensor:
    """Float grayscale thumbnail (size, size) in [0, 1] of a uint8 image."""
    gray = image.float().mean(dim=0, keepdim=True).unsqueeze(0) / 255.0
    return F.interpolate(gray, size=(size, size), mode="area")[0, 0]
