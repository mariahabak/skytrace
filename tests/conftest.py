"""Shared fixtures: small synthetic data, no real dataset needed.

Author: Mariah Habak
"""

import pytest
import torch

from dronebird.data.crops import CropBank
from dronebird.fake import make_fake_dataset


def make_bank(n: int = 24, canvas: int = 96, seed: int = 0) -> CropBank:
    """Birds are bright squares, drones dark bars, sizes from 10 to 90 px."""
    g = torch.Generator().manual_seed(seed)
    images = torch.full((n, 3, canvas, canvas), 150, dtype=torch.uint8)
    labels = torch.arange(n) % 2
    c = canvas // 2
    for i in range(n):
        if labels[i] == 0:
            images[i, :, c - 10:c + 10, c - 10:c + 10] = 240
        else:
            images[i, :, c - 4:c + 4, c - 25:c + 25] = 20
    images = (images.float() + torch.randint(0, 8, images.shape, generator=g)).clamp(0, 255)
    sizes = torch.linspace(10, 90, n)
    return CropBank(images.to(torch.uint8), labels, sizes,
                    [f"img{i}" for i in range(n)], [i // 2 for i in range(n)])


@pytest.fixture
def bank():
    return make_bank()


@pytest.fixture(scope="session")
def fake_root(tmp_path_factory):
    return make_fake_dataset(tmp_path_factory.mktemp("fake"))


EVALUATION = {"synthetic_min_size": 64, "synthetic_sizes": [48, 32, 24, 16],
              "real_max_size": 48,
              "real_size_bins": {"48": [36, 48], "32": [28, 36], "24": [20, 28], "16": [0, 20]},
              "test_degradation": {"jpeg_quality": 50, "shot_noise_photons": 0, "noise_seed": 0}}
