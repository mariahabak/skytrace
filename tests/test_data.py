"""Labels, crops, near-duplicate groups and the split.

Author: Mariah Habak
"""

import pytest
import torch

from dronebird.data.crops import CropMaker, CropSettings
from dronebird.data.duplicates import NearDuplicateGrouper
from dronebird.data.split import GroupSplitter, check_disjoint
from dronebird.data.yolo import Box, LabelIndex, parse_label_line

CLASS_IDS = {"0": "bird", "1": "drone"}


def test_label_lines_box_polygon_and_invalid():
    assert parse_label_line("1 0.5 0.4 0.2 0.1") == (1, 0.5, 0.4, 0.2, 0.1)
    cls, cx, cy, w, h = parse_label_line("0 0.1 0.2 0.3 0.2 0.3 0.6 0.1 0.6")
    assert (cls, cx, cy) == (0, pytest.approx(0.2), pytest.approx(0.4))
    assert (w, h) == (pytest.approx(0.2), pytest.approx(0.4))
    assert parse_label_line("") is None
    assert parse_label_line("0 0.5 0.5 0 0.1") is None


def test_boxes_dictionary_and_folder_check(fake_root):
    index = LabelIndex.build(fake_root, CLASS_IDS, image_size=160)
    boxes = index.boxes  # image path -> list of Box
    assert len(boxes) == 192
    assert all(isinstance(b, Box) for v in boxes.values() for b in v)
    index.check_folders()  # BT/DT folders agree with the class ids
    swapped = LabelIndex.build(fake_root, {"0": "drone", "1": "bird"}, image_size=160)
    with pytest.raises(ValueError):
        swapped.check_folders()


def test_crop_is_square_centred_and_padded_at_the_border():
    maker = CropMaker(CropSettings(margin=1.5, canvas=96, min_side=8))
    image = torch.zeros(3, 100, 100, dtype=torch.uint8)
    image[:, 2:12, 2:22] = 255  # a 20 x 10 object near the top-left corner
    crop, size = maker.cut(image, Box(0, 0.12, 0.07, 0.20, 0.10))
    assert crop.shape == (3, 96, 96) and crop.dtype == torch.uint8
    assert size == pytest.approx(20)
    # The window (30 px) leaves the frame, so the border is repeated; the
    # object (longest side 20 px -> 64 canvas px) sits in the centre.
    assert crop[0, 48, 48] == 255 and crop[0, 2, 2] == 0 and crop[0, 48, 2] == 0


def test_near_duplicates_are_grouped_and_sources_joined():
    g = torch.Generator().manual_seed(0)
    a = torch.rand(3, 16, 16, generator=g)
    b = torch.rand(3, 16, 16, generator=g)
    thumbs = torch.stack([a, a + 0.005, b, b.flip(-1), torch.rand(3, 16, 16, generator=g)])
    grouper = NearDuplicateGrouper(max_rms_difference=0.03)
    groups = grouper.group(thumbs, ["", "", "", "", ""])
    assert groups[0] == groups[1] and groups[0] != groups[2]
    assert groups[2] != groups[3]  # a flipped image is a different picture
    joined = grouper.group(thumbs, ["", "", "v7", "", "v7"])  # same source id in the name
    assert joined[2] == joined[4]


def test_split_keeps_groups_together_and_sets_are_disjoint():
    groups = [i // 3 for i in range(90)]  # 30 groups of 3 images
    labels = [(i // 3) % 2 for i in range(90)]
    splitter = GroupSplitter({"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)
    assignment = splitter.split(groups, labels)
    partition = [assignment[g] for g in groups]
    report = check_disjoint(partition, {"group": groups, "image": list(range(90))})
    assert sum(report["group"].values()) == 30
    assert GroupSplitter({"train": 0.7, "val": 0.15, "test": 0.15}, seed=1).split(
        groups, labels) == assignment
    with pytest.raises(RuntimeError):
        check_disjoint(["train", "test"], {"group": [5, 5]})
