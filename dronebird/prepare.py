"""Data preparation: labels -> near-duplicate groups -> split -> crops.

Author: Mariah Habak

Usage:  python -m dronebird.prepare configs/data.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dronebird import plots
from dronebird.config import Config
from dronebird.data.crops import CropBank, CropMaker, CropSettings
from dronebird.data.duplicates import NearDuplicateGrouper, group_report
from dronebird.data.images import colour_thumbnail, load_rgb
from dronebird.data.split import (PARTITIONS, GroupSplitter, check_disjoint,
                                  save_split)
from dronebird.data.yolo import CLASS_NAMES, LabelIndex


class DataPreparer:
    """Runs every preparation step from one data configuration."""

    def __init__(self, config: Config):
        self.__d = config.section("data")
        self.__out = Path(self.__d["processed_dir"])
        self.__crop_maker = CropMaker(CropSettings(**self.__d["crops"]))

    def run(self) -> dict:
        d = self.__d
        index = LabelIndex.build(d["dataset_root"], d["class_ids"],
                                 d.get("source_pattern", ""),
                                 d.get("min_box_pixels", 0), d.get("image_size", 640))
        index.check_folders()
        boxes = self.__subset(index.boxes, index, d.get("max_images_per_class"),
                              d["split_seed"])
        images = sorted(boxes)
        source_of = {img: src for src, imgs in index.sources.items() for img in imgs}
        print(f"{len(images)} labelled images; reading frames ...", flush=True)

        thumbs, crops = self.__read_all(images, boxes)
        grouper = NearDuplicateGrouper(d["duplicates"]["max_rms_difference"])
        groups = grouper.group(torch.stack(thumbs),
                               [source_of.get(img, "") for img in images])
        image_labels = [index.image_label(img) for img in images]
        assignment = GroupSplitter(d["split"], seed=d["split_seed"]).split(
            groups, image_labels)
        partition_of_image = [assignment[g] for g in groups]

        frames_by_group = {}  # group -> list of images
        for image, group in zip(images, groups):
            frames_by_group.setdefault(group, []).append(image)
        disjoint = check_disjoint(partition_of_image, {
            "image": images, "group": groups,
            "source": [source_of.get(img, "") for img in images]})

        banks = self.__banks(images, groups, partition_of_image, crops)
        self.__out.mkdir(parents=True, exist_ok=True)
        for part, bank in banks.items():
            bank.save(self.__out / f"{part}.pt")
        save_split([{"image": img, "partition": p, "group": g,
                     "source": source_of.get(img, ""), "label": CLASS_NAMES[lab],
                     "boxes": len(boxes[img])}
                    for img, p, g, lab in zip(images, partition_of_image, groups,
                                              image_labels)],
                   self.__out / "split.json")
        summary = {
            "label_problems": index.problems,
            "folder_check": index.folder_check,
            "near_duplicates": {**group_report(groups),
                                "confirmed_pairs": grouper.confirmed_pairs,
                                "groups_from_sources": len(index.sources)},
            "largest_groups": sorted((len(v) for v in frames_by_group.values()),
                                     reverse=True)[:10],
            "disjoint_check": disjoint,
            "partitions": {p: self.__describe(b) for p, b in banks.items()},
        }
        with open(self.__out / "summary.json", "w", encoding="utf-8") as h:
            json.dump(summary, h, indent=2)
        plots.data_overview(banks, self.__out / "figures", d, self.__crop_maker.settings)
        plots.duplicate_examples(images, groups, self.__out / "figures"
                                 / "near_duplicates.png")
        return summary

    @staticmethod
    def __subset(boxes: dict, index: LabelIndex, per_class, seed: int) -> dict:
        if not per_class:
            return boxes
        rng = np.random.default_rng(seed)
        by_class = {}
        for image in sorted(boxes):
            by_class.setdefault(index.image_label(image), []).append(image)
        keep = []
        for cls in sorted(by_class):
            chosen = by_class[cls]
            if len(chosen) > per_class:
                chosen = [chosen[i] for i in sorted(rng.choice(len(chosen), per_class,
                                                                replace=False))]
            keep += chosen
        return {img: boxes[img] for img in keep}

    def __read_all(self, images: list, boxes: dict) -> tuple:
        size = self.__d["duplicates"]["thumbnail_size"]
        thumbs, crops = [], []
        for n, image_path in enumerate(images, 1):
            image = load_rgb(image_path)
            thumbs.append(colour_thumbnail(image, size))
            crops.append([(*self.__crop_maker.cut(image, box), box.label)
                          for box in boxes[image_path]])
            if n % 1000 == 0:
                print(f"  {n} / {len(images)}", flush=True)
        return thumbs, crops

    def __banks(self, images, groups, partitions, crops) -> dict:
        lists = {p: ([], [], [], [], []) for p in PARTITIONS}
        for image, group, part, image_crops in zip(images, groups, partitions, crops):
            for crop, size, label in image_crops:
                pixels, labels, sizes, ids, gids = lists[part]
                pixels.append(crop)
                labels.append(label)
                sizes.append(size)
                ids.append(image)
                gids.append(group)
        canvas = self.__crop_maker.settings.canvas
        return {p: CropBank.from_lists(*lists[p], canvas=canvas) for p in PARTITIONS}

    def __describe(self, bank: CropBank) -> dict:
        ev = self.__d["evaluation"]
        size = bank.object_size
        labels = bank.labels
        counts = {CLASS_NAMES[c]: int((labels == c).sum()) for c in range(2)}
        synthetic = size >= ev["synthetic_min_size"]
        small = size < ev["real_max_size"]
        return {"crops": counts, "images": len(set(bank.image_ids)),
                "groups": len(set(bank.group_ids)),
                "synthetic_test_crops": {CLASS_NAMES[c]: int((synthetic & (labels == c)).sum())
                                         for c in range(2)},
                "real_small_crops": {CLASS_NAMES[c]: int((small & (labels == c)).sum())
                                     for c in range(2)},
                "object_size_median_px": float(size.median()) if len(size) else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    summary = DataPreparer(Config.from_file(args.config)).run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
