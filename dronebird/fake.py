"""A tiny fake dataset in the same YOLO layout, for tests and the smoke run.

Author: Mariah Habak

Usage:  python -m dronebird.fake data/smoke/raw

Each fake "video" has its own sky; its frames are almost identical, so
the near-duplicate grouping has something to find. Birds are drawn as
two dark wings, drones as a cross with four rotors. Object sizes go
from 6 to 70 px in 160 x 160 frames.
"""

import argparse
import math
from pathlib import Path

import torch
from torchvision.io import write_jpeg

SPLITS = {"train": 10, "valid": 3, "test": 3}  # videos per class
FRAMES = 6
SIZE = 160


def _sky(generator) -> torch.Tensor:
    top = torch.rand(3, generator=generator) * 0.4 + 0.5
    bottom = torch.rand(3, generator=generator) * 0.3 + 0.6
    t = torch.linspace(0, 1, SIZE).view(1, SIZE, 1)
    sky = top.view(3, 1, 1) * (1 - t) + bottom.view(3, 1, 1) * t
    return sky.expand(3, SIZE, SIZE).clone()


def _draw(image, label: int, cx: float, cy: float, size: float, phase: float):
    yy, xx = torch.meshgrid(torch.arange(SIZE).float(), torch.arange(SIZE).float(),
                            indexing="ij")
    dx, dy = (xx - cx) / (size / 2), (yy - cy) / (size / 2)
    if label == 0:  # bird: two wings that flap
        lift = 0.4 * math.sin(phase)
        wing = ((dy + lift * dx.abs() - 0.0).abs() < 0.18) & (dx.abs() < 1.0)
        body = (dx ** 2 / 0.05 + dy ** 2 / 0.04) < 1
        mask = wing | body
        colour = torch.tensor([0.15, 0.12, 0.10])
    else:  # drone: cross with rotors
        arms = (((dx - dy).abs() < 0.15) | ((dx + dy).abs() < 0.15)) & (dx.abs() < 0.8)
        rotors = torch.zeros_like(dx, dtype=torch.bool)
        for sx in (-0.7, 0.7):
            for sy in (-0.7, 0.7):
                rotors |= ((dx - sx) ** 2 + (dy - sy) ** 2) < 0.09
        mask = arms | rotors
        colour = torch.tensor([0.25, 0.25, 0.28])
    image[:, mask] = colour.view(3, 1)
    return image


def make_fake_dataset(root, seed: int = 0) -> Path:
    root = Path(root)
    g = torch.Generator().manual_seed(seed)
    for split, videos in SPLITS.items():
        for label, folder in ((0, "BT"), (1, "DT")):
            (root / split / "images" / folder).mkdir(parents=True, exist_ok=True)
            (root / split / "labels" / folder).mkdir(parents=True, exist_ok=True)
            for v in range(videos):
                sky = _sky(g)
                size = float(6 + torch.rand(1, generator=g) * 64)
                cx = float(30 + torch.rand(1, generator=g) * 100)
                cy = float(30 + torch.rand(1, generator=g) * 100)
                for f in range(FRAMES):
                    image = sky.clone()
                    image = _draw(image, label, cx + f, cy, size, phase=f)
                    image += torch.randn(image.shape, generator=g) * 0.01
                    name = f"{folder}_{split}_v{v:02d}_f{f:02d}"
                    write_jpeg((image.clamp(0, 1) * 255).to(torch.uint8),
                               str(root / split / "images" / folder / f"{name}.jpg"),
                               quality=95)
                    (root / split / "labels" / folder / f"{name}.txt").write_text(
                        f"{label} {(cx + f) / SIZE:.5f} {cy / SIZE:.5f} "
                        f"{size / SIZE:.5f} {size * 0.8 / SIZE:.5f}\n")
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    print(make_fake_dataset(args.root))


if __name__ == "__main__":
    main()
