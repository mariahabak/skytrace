"""Reading the YOLO labels of the Bird vs Drone dataset into dictionaries.

Author: Mariah Habak

Data: Srivastav, Shandilya, Datta, Yemets and Nagar (2023), "Segmented
Dataset Based on YOLOv7 for Drone vs. Bird Identification for Deep and
Machine Learning Algorithms", Mendeley Data, DOI 10.17632/6ghdz52pd7.
Licence CC BY 4.0.

Each image has a text file with one line per object:
    class cx cy w h            (box, values in [0, 1])
or  class x1 y1 x2 y2 ...      (polygon; we take its bounding box)

The main structure is a dictionary  image path -> list of Box.
"""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_NAMES = ("bird", "drone")


@dataclass(frozen=True)
class Box:
    """One labelled object, in fractions of the image width and height."""

    label: int  # 0 = bird, 1 = drone
    cx: float
    cy: float
    w: float
    h: float


def parse_label_line(line: str):
    """Return (class id, cx, cy, w, h) or None for an empty/invalid line."""
    parts = line.split()
    if len(parts) < 5:
        return None
    class_id = int(float(parts[0]))
    values = [float(v) for v in parts[1:]]
    if len(values) == 4:
        cx, cy, w, h = values
    else:  # polygon x1 y1 x2 y2 ...
        xs, ys = values[0::2], values[1::2]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    return class_id, cx, cy, w, h


def folder_class(relative: Path):
    """Class suggested by the folder names (BT = bird, DT = drone), or None."""
    hints = set()
    for part in relative.parts[:-1]:
        upper = part.upper()
        if upper == "BT" or "BIRD" in upper:
            hints.add("bird")
        if upper == "DT" or "DRONE" in upper:
            hints.add("drone")
    return hints.pop() if len(hints) == 1 else None


def _match_key(relative: Path, folder: str) -> str:
    """Path with the 'images'/'labels' folder and the suffix removed."""
    parts = ["*" if p.lower() == folder else p for p in relative.parts]
    return str(Path(*parts).with_suffix(""))


class LabelIndex:
    """Boxes of every labelled image under a dataset folder.

    ``boxes``:     dict  image path -> list of Box
    ``sources``:   dict  source id -> list of image paths (from the file
                   name, when ``source_pattern`` is given)
    ``problems``:  Counter of skipped files and why
    """

    def __init__(self, boxes: dict, sources: dict, problems: Counter,
                 folder_mismatch: int, folder_checked: int):
        self.__boxes = boxes
        self.__sources = sources
        self.__problems = problems
        self.__mismatch = folder_mismatch
        self.__checked = folder_checked

    @classmethod
    def build(cls, root, class_ids: dict, source_pattern: str = "",
              min_box_pixels: float = 0.0, image_size: int = 640) -> "LabelIndex":
        root = Path(root)
        images = {}
        for path in root.rglob("*"):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                images[_match_key(path.relative_to(root), "images")] = path
        boxes, problems = {}, Counter()
        mismatch = checked = 0
        for label_path in sorted(root.rglob("*.txt")):
            relative = label_path.relative_to(root)
            if not any(p.lower() == "labels" for p in relative.parts):
                continue  # README.txt and similar
            image = images.get(_match_key(relative, "labels"))
            if image is None:
                problems["label without image"] += 1
                continue
            found = []
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parsed = parse_label_line(line)
                if parsed is None:
                    continue
                class_id, cx, cy, w, h = parsed
                name = class_ids.get(str(class_id))
                if name is None:
                    problems[f"unknown class id {class_id}"] += 1
                    continue
                if max(w, h) * image_size < min_box_pixels:
                    problems["box too small"] += 1
                    continue
                found.append(Box(CLASS_NAMES.index(name), cx, cy, w, h))
            if not found:
                problems["no usable box"] += 1
                continue
            hint = folder_class(image.relative_to(root))
            if hint is not None:
                checked += 1
                majority = Counter(b.label for b in found).most_common(1)[0][0]
                mismatch += CLASS_NAMES[majority] != hint
            boxes[str(image)] = found
        problems["image without label"] = max(0, len(images) - len(boxes)
                                               - problems["no usable box"])
        sources = cls.__group_sources(boxes, source_pattern)
        return cls(boxes, sources, problems, mismatch, checked)

    @staticmethod
    def __group_sources(boxes: dict, pattern: str) -> dict:
        sources = {}
        if not pattern:
            return sources
        regex = re.compile(pattern)
        for image in boxes:
            match = regex.search(Path(image).stem)
            if match:
                key = match.group(1) if match.groups() else match.group(0)
                sources.setdefault(key, []).append(image)
        return sources

    def check_folders(self, tolerance: float = 0.01) -> None:
        """Fail if the class ids disagree with the BT/DT folder names."""
        if self.__checked and self.__mismatch / self.__checked > tolerance:
            raise ValueError(
                f"{self.__mismatch} of {self.__checked} images have a class id "
                "that disagrees with their BT/DT folder: swap 'class_ids' in "
                "configs/data.json (run python -m dronebird.audit first)")

    @property
    def boxes(self) -> dict:
        return {k: list(v) for k, v in self.__boxes.items()}

    @property
    def sources(self) -> dict:
        return {k: list(v) for k, v in self.__sources.items()}

    @property
    def problems(self) -> dict:
        return dict(self.__problems)

    @property
    def folder_check(self) -> dict:
        return {"checked": self.__checked, "mismatch": self.__mismatch}

    def image_label(self, image: str) -> int:
        """Majority class of the boxes in one image."""
        return Counter(b.label for b in self.__boxes[image]).most_common(1)[0][0]

    def __len__(self) -> int:
        return len(self.__boxes)
