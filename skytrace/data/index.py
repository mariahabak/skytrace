"""Building a table of the videos we use, with their labels and sessions.

Authors: Maria Habak, [Partner name]

Data: Svanström, Alonso-Fernandez and Englund (2021), "A dataset for
multi-sensor drone detection", Data in Brief 39, 107521. Licence CC0-1.0.
Labels are MATLAB groundTruth objects, read with the third-party
``mcos-decoder`` package (https://pypi.org/project/mcos-decoder/).
"""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

SENSOR_FOLDERS = {"IR": "Video_IR", "V": "Video_V"}
SPREADSHEET = "Video_dataset_description.xlsx"
COLUMNS = ["sensor", "cls", "number", "distance", "inter_bin", "drone_type"]


def read_boxes(label_path) -> list:
    """Per-frame boxes ``(x, y, w, h)`` in pixels, or None when absent."""
    from mcos_decoder import load_groundtruth

    return load_groundtruth(str(label_path))


def read_creation_time(video_path) -> str:
    """The ``creation_time`` tag of an mp4 file, or '' if missing."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format_tags=creation_time",
         "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip()


class VideoIndex:
    """One row per video of the chosen sensor and classes.

    Each row keeps the file paths, the class, the distance bin given by
    the dataset authors, how many frames carry a box, and the file's
    creation time, which we use to rebuild recording sessions.
    """

    def __init__(self, table: pd.DataFrame):
        self.__table = table.reset_index(drop=True)

    @classmethod
    def build(cls, dataset_root, sensor: str, classes: list,
              min_labelled_frames: int) -> "VideoIndex":
        root = Path(dataset_root) / "Data"
        sheet = pd.read_excel(root / SPREADSHEET)
        sheet.columns = COLUMNS
        sheet = sheet[(sheet.sensor == sensor) & (sheet.cls.isin(classes))]

        rows = []
        for entry in sheet.itertuples():
            stem = f"{sensor}_{entry.cls}_{int(entry.number):03d}"
            folder = root / SENSOR_FOLDERS[sensor]
            video = folder / f"{stem}.mp4"
            labels = folder / f"{stem}_LABELS.mat"
            boxes = read_boxes(labels)
            labelled = [b for b in boxes if b is not None]
            sides = [max(b[2], b[3]) for b in labelled]
            rows.append({
                "video_id": stem,
                "cls": entry.cls,
                "distance": entry.distance,
                "drone_type": "" if pd.isna(entry.drone_type) else entry.drone_type,
                "video_path": str(video),
                "label_path": str(labels),
                "labelled_frames": len(labelled),
                "median_box_side": float(np.median(sides)) if sides else np.nan,
                "creation_time": read_creation_time(video),
            })
        table = pd.DataFrame(rows)
        table = table[table.labelled_frames >= min_labelled_frames]
        return cls(table)

    @classmethod
    def from_csv(cls, path) -> "VideoIndex":
        return cls(pd.read_csv(path, keep_default_na=False,
                               na_values={"median_box_side": [""]}))

    def to_csv(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.__table.to_csv(path, index=False)

    @property
    def table(self) -> pd.DataFrame:
        return self.__table.copy()

    def __len__(self) -> int:
        return len(self.__table)
