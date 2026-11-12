"""DistanceAugment and the test degradations.

Author: Mariah Habak
"""

import pytest
import torch

from dronebird.augment import (BicubicShrink, DistanceAugment, JpegCompression,
                               gaussian_kernel, small_side, synthetic_test_degradation)
from dronebird.data.datasets import IMAGENET_MEAN, IMAGENET_STD, normalise

TEST = {"jpeg_quality": 50}


def _augment(p):
    return DistanceAugment(p=p, min_size=16, max_size=16, margin=1.5, canvas=96,
                           blur_sigma_max=1.0, noise_std_max=0.04)


def test_distance_augment_keeps_shape_and_range(bank):
    x = bank.images[:1].float() / 255
    assert torch.equal(_augment(0.0)(x), x)
    torch.manual_seed(0)
    y = _augment(1.0)(x)
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1
    assert not torch.allclose(y, x)


def test_test_recipe_is_deterministic_and_differs_from_training():
    x = torch.rand(2, 3, 96, 96, generator=torch.Generator().manual_seed(0))
    recipe = synthetic_test_degradation(16, TEST, margin=1.5, canvas=96)
    assert torch.equal(recipe(x), recipe(x))  # the same inputs for every model
    torch.manual_seed(0)
    trained_on = _augment(1.0)(x)
    assert (recipe(x) - trained_on).abs().mean() > 0.01


def test_shrink_sizes_and_jpeg():
    x = torch.rand(2, 3, 96, 96)
    assert small_side(16, 1.5) == 24
    assert BicubicShrink(16, 1.5)(x).shape == (2, 3, 24, 24)
    assert JpegCompression(50)(x).shape == x.shape
    k = gaussian_kernel(1.0, 3)
    assert k.shape == (3, 1, 7, 7) and k[0].sum().item() == pytest.approx(1.0)


def test_normalisation_by_broadcasting():
    x = torch.rand(5, 3, 8, 8)
    expected = torch.stack([(x[:, c] - IMAGENET_MEAN[c, 0, 0]) / IMAGENET_STD[c, 0, 0]
                            for c in range(3)], dim=1)
    assert torch.allclose(normalise(x), expected, atol=1e-6)
