"""Look at the dataset before preparing it.

Author: Mariah Habak

Usage:  python -m dronebird.audit data/raw/bird_vs_drone

Prints the folder structure, the label format, which class id appears in
the bird (BT) and drone (DT) folders, the object sizes in pixels and
sample file names. Use it to set ``class_ids`` and ``source_pattern`` in
configs/data.json. Saves the same information to audit.json.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from dronebird.data.images import load_rgb
from dronebird.data.yolo import IMAGE_SUFFIXES, folder_class, parse_label_line


class DatasetAudit:
    def __init__(self, root, sample_images: int = 200, seed: int = 0):
        self.__root = Path(root)
        self.__sample = sample_images
        self.__rng = np.random.default_rng(seed)

    def run(self) -> dict:
        root = self.__root
        folders = Counter()
        images, labels = [], []
        for path in root.rglob("*"):
            if path.is_file():
                folders[str(path.parent.relative_to(root))] += 1
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    images.append(path)
                elif path.suffix.lower() == ".txt" and any(
                        p.lower() == "labels" for p in path.relative_to(root).parts):
                    labels.append(path)

        formats, ids_by_folder, boxes_per_file, sizes = Counter(), {}, Counter(), []
        for label in labels:
            hint = folder_class(label.relative_to(root)) or "unknown folder"
            n = 0
            for line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if not parts:
                    continue
                formats["box (5 values)" if len(parts) == 5 else
                        f"polygon/other ({len(parts)} values)"] += 1
                parsed = parse_label_line(line)
                if parsed:
                    n += 1
                    ids_by_folder.setdefault(hint, Counter())[str(parsed[0])] += 1
                    sizes.append(max(parsed[3], parsed[4]))
            boxes_per_file[min(n, 5)] += 1

        shapes = Counter()
        for i in self.__rng.choice(len(images), min(self.__sample, len(images)), replace=False):
            shapes[tuple(load_rgb(images[i]).shape[1:])] += 1
        width = shapes.most_common(1)[0][0][1] if shapes else 640
        px = np.asarray(sizes) * width
        names = {}
        for image in images:
            hint = folder_class(image.relative_to(root)) or "unknown folder"
            names.setdefault(hint, [])
            if len(names[hint]) < 15:
                names[hint].append(image.name)
        report = {
            "root": str(root),
            "folders (files per folder)": dict(sorted(folders.items())),
            "images": len(images), "label files": len(labels),
            "label line formats": dict(formats),
            "class ids per folder (BT = bird, DT = drone)": {k: dict(v) for k, v in
                                                             ids_by_folder.items()},
            "boxes per label file (5 = 5 or more)": dict(sorted(boxes_per_file.items())),
            "image sizes in a sample (H, W)": {str(k): v for k, v in shapes.items()},
            "object size px (longest box side)": {
                f"p{q}": round(float(np.percentile(px, q)), 1) for q in (1, 5, 25, 50, 75, 95, 99)}
            if len(px) else {},
            "objects < 48 px": int((px < 48).sum()), "objects >= 64 px": int((px >= 64).sum()),
            "sample file names": names,
        }
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root")
    args = parser.parse_args()
    report = DatasetAudit(args.root).run()
    text = json.dumps(report, indent=2)
    Path("audit.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
