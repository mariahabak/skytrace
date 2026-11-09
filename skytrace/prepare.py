"""Data preparation: index -> sessions -> group split -> clips -> plots.

Authors: Maria Habak, [Partner name]

Usage:  python -m skytrace.prepare configs/data.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from skytrace.config import Config
from skytrace.data.clips import ClipExtractor, ClipSet, ClipSettings
from skytrace.data.index import VideoIndex
from skytrace.data.split import (PARTITIONS, GroupSplitter, assign_sessions,
                                 save_split)
from skytrace import plots


class DataPreparer:
    """Runs every preparation step from one data configuration."""

    def __init__(self, config: Config):
        self.__data = config.section("data")
        self.__out = Path(self.__data["processed_dir"])

    def run(self) -> dict:
        d = self.__data
        index = VideoIndex.build(d["dataset_root"], d["sensor"], d["classes"],
                                 d["min_labelled_frames"])
        table = index.table
        table["session"] = assign_sessions(table, d["session_gap_minutes"])

        splitter = GroupSplitter(d["split"], seed=d["split_seed"])
        assignment = splitter.split(table, weight_column="labelled_frames")
        table["partition"] = table.video_id.map(assignment)
        self.__check_no_leakage(table)

        self.__out.mkdir(parents=True, exist_ok=True)
        table.to_csv(self.__out / "index.csv", index=False)
        save_split(assignment, dict(zip(table.video_id, table.session)),
                   self.__out / "split.json")

        extractor = ClipExtractor(ClipSettings(**d["clips"]))
        sets = {}
        for part in PARTITIONS:
            rows = table[table.partition == part]
            sets[part] = ClipSet.from_videos(extractor, rows)
            sets[part].save(self.__out / f"{part}.pt")

        summary = self.__summary(table, sets)
        with open(self.__out / "summary.json", "w", encoding="utf-8") as h:
            json.dump(summary, h, indent=2)
        plots.data_overview(table, sets, self.__out / "figures")
        return summary

    @staticmethod
    def __check_no_leakage(table: pd.DataFrame) -> None:
        per_session = table.groupby("session").partition.nunique()
        if (per_session > 1).any():
            raise RuntimeError("A session was split across partitions")

    @staticmethod
    def __summary(table: pd.DataFrame, sets: dict) -> dict:
        out = {}
        for part, clip_set in sets.items():
            rows = table[table.partition == part]
            labels = clip_set.labels.tolist()
            out[part] = {
                "videos": rows.groupby("cls").size().to_dict(),
                "sessions": rows.groupby("cls").session.nunique().to_dict(),
                "clips": {"BIRD": labels.count(0), "DRONE": labels.count(1)},
            }
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()
    summary = DataPreparer(Config.from_file(args.config)).run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
