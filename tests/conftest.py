"""Shared test fixtures: small synthetic clip sets (no dataset needed).

Authors: Maria Habak, [Partner name]
"""

import numpy as np
import pytest
import torch

from skytrace.data.clips import ClipSet


def make_clip_set(n: int = 40, length: int = 8, size: int = 32, seed: int = 0) -> ClipSet:
    """Birds are small bright blobs that flicker; drones are wide steady bars."""
    rng = np.random.default_rng(seed)
    clips = np.full((n, length, size, size), 90, dtype=np.uint8)
    labels = np.arange(n) % 2
    tracks = np.zeros((n, length, 4), dtype=np.float32)
    c = size // 2
    for i in range(n):
        for t in range(length):
            if labels[i] == 0:
                r = 2 + (t % 2)
                clips[i, t, c - r:c + r, c - r:c + r] = 200
                tracks[i, t] = [0.5 + 0.01 * t, 0.5 + 0.01 * (t % 2), 0.03, 0.03]
            else:
                clips[i, t, c - 1:c + 2, c - 6:c + 6] = 220
                tracks[i, t] = [0.5 + 0.01 * t, 0.5, 0.06, 0.02]
        clips[i] = np.clip(clips[i] + rng.integers(0, 10, clips[i].shape), 0, 255)
    box_side = np.where(labels == 0, 8.0, 15.0).astype(np.float32)
    return ClipSet(torch.from_numpy(clips), torch.from_numpy(labels).long(),
                   torch.from_numpy(tracks), torch.from_numpy(box_side),
                   [f"V{i // 4}" for i in range(n)], [0] * n)


@pytest.fixture
def clip_set():
    return make_clip_set()
