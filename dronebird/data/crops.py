"""Cutting object crops out of the frames, the same way for every model.

Author: Mariah Habak

The crop pipeline, step by step (shapes for the default settings):
1. Box in pixels: centre (cx, cy), width w, height h; object size
   L = max(w, h).
2. Square window around the centre with side = margin x L
   (margin 1.5), so the aspect ratio of the object is never changed
   and the sky around it is kept.
3. Where the window leaves the frame, the frame is padded by
   repeating its border pixels (no black bars, which a model could use
   as a cue).
4. The window is cut with tensor slicing:
       crop = padded[:, top:top + side, left:left + side]   (3, side, side)
5. ``to_canvas`` resizes it to (3, 96, 96) with bilinear interpolation.
   The object's longest side is then 96 / 1.5 = 64 canvas pixels.
   This is what we call "full size".
The crops are stored as uint8 tensors (N, 3, 96, 96).
"""

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from dronebird.data.images import to_canvas


@dataclass(frozen=True)
class CropSettings:
    margin: float
    canvas: int
    min_side: int

    @property
    def full_object_size(self) -> float:
        """Object size in canvas pixels for an unshrunk crop."""
        return self.canvas / self.margin


class CropMaker:
    def __init__(self, settings: CropSettings):
        self.__s = settings

    @property
    def settings(self) -> CropSettings:
        return self.__s

    def window(self, box, width: int, height: int) -> tuple:
        """(left, top, side, object_size) of the square window, in pixels."""
        cx, cy = box.cx * width, box.cy * height
        w, h = box.w * width, box.h * height
        object_size = max(w, h)
        side = max(self.__s.min_side, int(math.ceil(self.__s.margin * object_size)))
        left = int(round(cx - side / 2))
        top = int(round(cy - side / 2))
        return left, top, side, object_size

    def cut(self, image: torch.Tensor, box) -> tuple:
        """uint8 image (3, H, W) -> uint8 crop (3, canvas, canvas), object size."""
        _, height, width = image.shape
        left, top, side, object_size = self.window(box, width, height)
        pad_left, pad_top = max(0, -left), max(0, -top)
        pad_right = max(0, left + side - width)
        pad_bottom = max(0, top + side - height)
        x = image.float().unsqueeze(0)
        if pad_left or pad_top or pad_right or pad_bottom:
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
        top, left = top + pad_top, left + pad_left
        crop = x[0, :, top:top + side, left:left + side]
        canvas = to_canvas(crop, self.__s.canvas)
        return canvas.round().clamp(0, 255).to(torch.uint8), object_size


class CropBank:
    """All crops of one partition.

    images:      uint8   (N, 3, S, S)
    labels:      int64   (N,)   0 = bird, 1 = drone
    object_size: float32 (N,)   longest box side in the original frame (px)
    image_ids, group_ids: where each crop comes from
    """

    def __init__(self, images, labels, object_size, image_ids, group_ids):
        self.__images = images
        self.__labels = labels
        self.__size = object_size
        self.__image_ids = list(image_ids)
        self.__group_ids = list(group_ids)
        n = len(labels)
        if not (len(images) == len(object_size) == len(self.__image_ids)
                == len(self.__group_ids) == n):
            raise ValueError("All CropBank fields need the same length")

    @classmethod
    def from_lists(cls, crops: list, labels: list, sizes: list, image_ids: list,
                   group_ids: list, canvas: int) -> "CropBank":
        images = torch.stack(crops) if crops else torch.zeros((0, 3, canvas, canvas),
                                                               dtype=torch.uint8)
        return cls(images, torch.tensor(labels, dtype=torch.int64),
                   torch.tensor(sizes, dtype=torch.float32), image_ids, group_ids)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"images": self.__images, "labels": self.__labels,
                    "object_size": self.__size, "image_ids": self.__image_ids,
                    "group_ids": self.__group_ids}, path)

    @classmethod
    def load(cls, path) -> "CropBank":
        d = torch.load(path, weights_only=False)
        return cls(d["images"], d["labels"], d["object_size"], d["image_ids"],
                   d["group_ids"])

    def select(self, mask: torch.Tensor) -> "CropBank":
        idx = torch.nonzero(mask, as_tuple=True)[0]
        return CropBank(self.__images[idx], self.__labels[idx], self.__size[idx],
                        [self.__image_ids[i] for i in idx.tolist()],
                        [self.__group_ids[i] for i in idx.tolist()])

    def __len__(self) -> int:
        return len(self.__labels)

    @property
    def images(self) -> torch.Tensor:
        return self.__images

    @property
    def labels(self) -> torch.Tensor:
        return self.__labels

    @property
    def object_size(self) -> torch.Tensor:
        return self.__size

    @property
    def image_ids(self) -> list:
        return list(self.__image_ids)

    @property
    def group_ids(self) -> list:
        return list(self.__group_ids)
