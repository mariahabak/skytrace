"""Splitting videos into train / validation / test by recording session.

Authors: Maria Habak, [Partner name]

The dataset has no session column. Files exported together share a
creation time, and bird and drone videos never share one, so we treat
videos of the same class whose creation times are close as one session.
A whole session always goes into one partition, so near-identical clips
of the same flight cannot appear in both training and test data.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PARTITIONS = ("train", "val", "test")


def assign_sessions(table: pd.DataFrame, gap_minutes: float) -> pd.Series:
    """Session label per video: same class, creation times within the gap."""
    times = pd.to_datetime(table["creation_time"], utc=True)
    sessions = pd.Series(index=table.index, dtype=object)
    for cls, group in table.assign(_t=times).sort_values("_t").groupby("cls"):
        new_session = group["_t"].diff() > pd.Timedelta(minutes=gap_minutes)
        numbers = new_session.cumsum()
        sessions.loc[group.index] = [f"{cls}_S{n:02d}" for n in numbers]
    return sessions


class GroupSplitter:
    """Assigns whole sessions to partitions, class by class.

    For each class it draws many random assignments of sessions to the
    three partitions (with a fixed seed) and keeps the one whose sizes
    are closest to the target fractions, with every partition non-empty.
    """

    def __init__(self, fractions: dict, seed: int, trials: int = 5000):
        total = sum(fractions[p] for p in PARTITIONS)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Split fractions must sum to 1")
        self.__fractions = np.array([fractions[p] for p in PARTITIONS])
        self.__seed = seed
        self.__trials = trials

    def split(self, table: pd.DataFrame, weight_column: str) -> dict:
        """Map each video_id to 'train', 'val' or 'test'."""
        rng = np.random.default_rng(self.__seed)
        result = {}
        for cls in sorted(table["cls"].unique()):
            rows = table[table["cls"] == cls]
            sizes = rows.groupby("session")[weight_column].sum()
            choice = self.__best_assignment(sizes.to_numpy(), rng)
            for session, part in zip(sizes.index, choice):
                for video in rows.loc[rows.session == session, "video_id"]:
                    result[video] = PARTITIONS[part]
        return result

    def __best_assignment(self, sizes: np.ndarray, rng) -> np.ndarray:
        if len(sizes) < len(PARTITIONS):
            raise ValueError("A class needs at least three sessions to split")
        target = self.__fractions * sizes.sum()
        best, best_cost = None, np.inf
        for _ in range(self.__trials):
            choice = rng.integers(0, len(PARTITIONS), size=len(sizes))
            totals = np.bincount(choice, weights=sizes, minlength=len(PARTITIONS))
            if (totals == 0).any():
                continue
            cost = (((totals - target) / target) ** 2).sum()
            if cost < best_cost:
                best, best_cost = choice, cost
        return best


def save_split(assignment: dict, sessions: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = [{"video_id": v, "partition": p, "session": sessions[v]}
            for v, p in sorted(assignment.items())]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=1)


def load_split(path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as handle:
        return pd.DataFrame(json.load(handle))
