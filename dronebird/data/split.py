"""Train / validation / test split by near-duplicate group.

Author: Mariah Habak

A group (frames from one video, or near-identical frames) always goes
to one partition. The split is made with a fixed seed, before any
training, and saved to disk. ``check_disjoint`` uses sets to prove that
no image, group or source id appears in two partitions.
"""

import json
from pathlib import Path

import numpy as np

PARTITIONS = ("train", "val", "test")


class GroupSplitter:
    """Assigns whole groups to partitions, class by class.

    Groups are shuffled with the seed, then each one goes to the
    partition that is furthest below its target share of images.
    """

    def __init__(self, fractions: dict, seed: int):
        total = sum(fractions[p] for p in PARTITIONS)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Split fractions must sum to 1")
        self.__fractions = {p: float(fractions[p]) for p in PARTITIONS}
        self.__seed = seed

    def split(self, groups: list, labels: list) -> dict:
        """Map each group id to a partition name."""
        rng = np.random.default_rng(self.__seed)
        members = {}  # group -> list of image indices
        for index, group in enumerate(groups):
            members.setdefault(group, []).append(index)
        by_class = {}  # class -> list of groups
        for group, indices in members.items():
            votes = np.bincount([labels[i] for i in indices], minlength=2)
            by_class.setdefault(int(votes.argmax()), []).append(group)

        assignment = {}
        for cls in sorted(by_class):
            class_groups = list(by_class[cls])
            rng.shuffle(class_groups)
            total = sum(len(members[g]) for g in class_groups)
            filled = {p: 0 for p in PARTITIONS}
            for group in class_groups:
                deficit = {p: self.__fractions[p] * total - filled[p] for p in PARTITIONS}
                target = max(PARTITIONS, key=lambda p: (deficit[p], self.__fractions[p]))
                assignment[group] = target
                filled[target] += len(members[group])
        return assignment


def check_disjoint(partition_of_image: list, keys: dict) -> dict:
    """Sets of every key per partition, and proof that they never overlap.

    ``keys`` maps a key name (e.g. "image", "group", "source") to one value
    per image. Raises if any value appears in two partitions.
    """
    report = {}
    for name, values in keys.items():
        sets = {p: set() for p in PARTITIONS}
        for partition, value in zip(partition_of_image, values):
            if value not in (None, ""):
                sets[partition].add(value)
        for i, a in enumerate(PARTITIONS):
            for b in PARTITIONS[i + 1:]:
                shared = sets[a] & sets[b]
                if shared:
                    raise RuntimeError(f"{len(shared)} {name} value(s) in both "
                                       f"{a} and {b}, e.g. {next(iter(shared))}")
        report[name] = {p: len(sets[p]) for p in PARTITIONS}
    return report


def save_split(rows: list, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=1)


def load_split(path) -> list:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
