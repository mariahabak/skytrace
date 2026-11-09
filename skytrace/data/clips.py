"""Cutting fixed-length clips of object crops out of the videos.

Authors: Maria Habak, [Partner name]

A clip is ``length`` frames taken every ``frame_step`` frames, all with
a labelled box. Each frame is cropped as a square around the box centre
(side = longest box side x ``context``), padded at the image border and
resized to ``crop_size`` x ``crop_size``. Infrared frames are grayscale,
so a clip is stored as uint8 with shape (length, crop_size, crop_size).
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from skytrace.data.index import read_boxes

CLASS_NAMES = ("BIRD", "DRONE")


@dataclass(frozen=True)
class ClipSettings:
    length: int
    frame_step: int
    window_stride: int
    crop_size: int
    context: float
    min_crop_side: int


def read_gray_frames(video_path) -> list:
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    capture.release()
    return frames


def crop_around(frame: np.ndarray, box, settings: ClipSettings) -> np.ndarray:
    x, y, w, h = box
    side = max(settings.min_crop_side, int(round(max(w, h) * settings.context)))
    cx, cy = x + w / 2.0, y + h / 2.0
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    pad = side
    padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    patch = padded[top + pad: top + pad + side, left + pad: left + pad + side]
    interp = cv2.INTER_AREA if side > settings.crop_size else cv2.INTER_LINEAR
    return cv2.resize(patch, (settings.crop_size, settings.crop_size),
                      interpolation=interp)


class ClipExtractor:
    """Turns one video and its boxes into a list of clips."""

    def __init__(self, settings: ClipSettings):
        self.__settings = settings

    @property
    def settings(self) -> ClipSettings:
        return self.__settings

    def window_starts(self, boxes: list) -> list:
        """First frame of every window whose frames all carry a box."""
        s = self.__settings
        span = (s.length - 1) * s.frame_step + 1
        starts = []
        for start in range(0, len(boxes) - span + 1, s.window_stride):
            frames = range(start, start + span, s.frame_step)
            if all(boxes[i] is not None for i in frames):
                starts.append(start)
        return starts

    def extract(self, video_path, label_path) -> dict:
        s = self.__settings
        frames = read_gray_frames(video_path)
        boxes = read_boxes(label_path)[: len(frames)]
        height, width = frames[0].shape
        clips, tracks, starts = [], [], []
        for start in self.window_starts(boxes):
            picked = range(start, start + (s.length - 1) * s.frame_step + 1,
                           s.frame_step)
            clips.append(np.stack([crop_around(frames[i], boxes[i], s)
                                   for i in picked]))
            tracks.append([[(boxes[i][0] + boxes[i][2] / 2) / width,
                            (boxes[i][1] + boxes[i][3] / 2) / height,
                            boxes[i][2] / width, boxes[i][3] / height]
                           for i in picked])
            starts.append(start)
        return {"clips": clips, "tracks": tracks, "starts": starts,
                "frame_size": (width, height)}


class ClipSet:
    """All clips of one partition, stored as tensors.

    clips:  uint8   (N, T, S, S)
    labels: int64   (N,)     0 = bird, 1 = drone
    tracks: float32 (N, T, 4) box centre x, y and size w, h, as a
            fraction of the frame
    box_side: float32 (N,)   median longest box side in pixels
    video_ids, starts: where each clip comes from
    """

    def __init__(self, clips, labels, tracks, box_side, video_ids, starts):
        self.__clips = clips
        self.__labels = labels
        self.__tracks = tracks
        self.__box_side = box_side
        self.__video_ids = list(video_ids)
        self.__starts = list(starts)
        if not (len(clips) == len(labels) == len(tracks) == len(box_side)
                == len(self.__video_ids) == len(self.__starts)):
            raise ValueError("All ClipSet fields need the same length")

    @classmethod
    def from_videos(cls, extractor: ClipExtractor, rows) -> "ClipSet":
        clips, labels, tracks, sides, videos, starts = [], [], [], [], [], []
        for row in rows.itertuples():
            out = extractor.extract(row.video_path, row.label_path)
            for clip, track, start in zip(out["clips"], out["tracks"], out["starts"]):
                clips.append(clip)
                labels.append(CLASS_NAMES.index(row.cls))
                tracks.append(track)
                t = np.asarray(track)
                w_px = t[:, 2] * out["frame_size"][0]
                h_px = t[:, 3] * out["frame_size"][1]
                sides.append(float(np.median(np.maximum(w_px, h_px))))
                videos.append(row.video_id)
                starts.append(start)
        s = extractor.settings
        empty = (0, s.length, s.crop_size, s.crop_size)
        return cls(
            torch.from_numpy(np.stack(clips)) if clips else torch.zeros(empty, dtype=torch.uint8),
            torch.tensor(labels, dtype=torch.int64),
            torch.tensor(np.asarray(tracks, dtype=np.float32).reshape(-1, s.length, 4)),
            torch.tensor(sides, dtype=torch.float32),
            videos, starts,
        )

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"clips": self.__clips, "labels": self.__labels,
                    "tracks": self.__tracks, "box_side": self.__box_side,
                    "video_ids": self.__video_ids, "starts": self.__starts}, path)

    @classmethod
    def load(cls, path) -> "ClipSet":
        d = torch.load(path, weights_only=False)
        return cls(d["clips"], d["labels"], d["tracks"], d["box_side"],
                   d["video_ids"], d["starts"])

    def subset(self, indices) -> "ClipSet":
        idx = torch.as_tensor(list(indices), dtype=torch.int64)
        return ClipSet(self.__clips[idx], self.__labels[idx], self.__tracks[idx],
                       self.__box_side[idx],
                       [self.__video_ids[i] for i in idx.tolist()],
                       [self.__starts[i] for i in idx.tolist()])

    def with_clips(self, clips) -> "ClipSet":
        """The same clips' metadata with new pixels (used for degradations)."""
        return ClipSet(clips, self.__labels, self.__tracks, self.__box_side,
                       self.__video_ids, self.__starts)

    def __len__(self) -> int:
        return len(self.__labels)

    @property
    def clips(self):
        return self.__clips.clone()

    @property
    def labels(self):
        return self.__labels.clone()

    @property
    def tracks(self):
        return self.__tracks.clone()

    @property
    def box_side(self):
        return self.__box_side.clone()

    @property
    def video_ids(self) -> list:
        return list(self.__video_ids)

    @property
    def starts(self) -> list:
        return list(self.__starts)
