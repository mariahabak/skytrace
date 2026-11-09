"""Clip windows and image degradations.

Authors: Maria Habak, [Partner name]
"""

import numpy as np
import pytest
import torch

from skytrace.data.clips import ClipExtractor, ClipSettings, crop_around
from skytrace.data.datasets import ClipAugmenter, ClipDataset
from skytrace.data.degrade import apply

SETTINGS = ClipSettings(length=4, frame_step=2, window_stride=7, crop_size=16,
                        context=2.0, min_crop_side=8)


def test_windows_skip_frames_without_a_box():
    boxes = [(1, 1, 2, 2)] * 20
    boxes[9] = None  # frame 9 is inside the window starting at 7 (7, 9, 11, 13)
    starts = ClipExtractor(SETTINGS).window_starts(boxes)
    assert starts == [0]  # window 7 is dropped; 14 would need frame 20


def test_crop_near_the_border_keeps_the_size():
    frame = np.arange(100, dtype=np.uint8).reshape(10, 10)
    patch = crop_around(frame, (0, 0, 3, 3), SETTINGS)
    assert patch.shape == (16, 16)


@pytest.mark.parametrize("kind,severity", [("blur", 2.0), ("resolution", 4),
                                           ("contrast", 0.5)])
def test_degradations_keep_shape_and_zero_is_identity(clip_set, kind, severity):
    clips = clip_set.clips
    out = apply(clips, kind, severity)
    assert out.shape == clips.shape and out.dtype == torch.uint8
    assert torch.equal(apply(clips, kind, 0), clips)
    # Every degradation removes detail: the frames become less varied.
    assert out.float().std() < clips.float().std()


def test_augmentation_is_the_same_on_every_frame():
    clip = torch.stack([torch.arange(256, dtype=torch.uint8).reshape(16, 16)] * 4)
    out = ClipAugmenter(0.1, 0.2, generator=torch.Generator().manual_seed(1))(clip)
    for t in range(1, 4):
        assert torch.equal(out[0], out[t])


def test_clip_dataset_shape(clip_set):
    x, y = ClipDataset(clip_set, input_size=24)[0]
    assert x.shape == (8, 3, 24, 24) and int(y) in (0, 1)
