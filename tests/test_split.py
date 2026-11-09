"""The group split must never put one recording session in two partitions.

Authors: Maria Habak, [Partner name]
"""

import pandas as pd
import pytest

from skytrace.data.split import GroupSplitter, assign_sessions


def _table():
    rows = []
    # Six bird sessions and six drone sessions, two hours apart,
    # each with three videos exported a few minutes apart.
    for cls in ("BIRD", "DRONE"):
        for s in range(6):
            for v in range(3):
                minute = s * 120 + v * 5
                rows.append({"video_id": f"{cls}_{s}_{v}", "cls": cls,
                             "creation_time": f"2020-03-20T{8 + minute // 60:02d}:"
                                              f"{minute % 60:02d}:00Z",
                             "labelled_frames": 300})
    return pd.DataFrame(rows)


def test_sessions_follow_time_gaps():
    table = _table()
    sessions = assign_sessions(table, gap_minutes=30)
    assert sessions.nunique() == 12
    # Videos of the same session share a label; classes never share one.
    for (cls, s), group in table.assign(sess=sessions).groupby(
            [table.cls, table.video_id.str.split("_").str[1]]):
        assert group.sess.nunique() == 1


def test_no_session_in_two_partitions_and_seed_is_repeatable():
    table = _table()
    table["session"] = assign_sessions(table, gap_minutes=30)
    splitter = GroupSplitter({"train": 0.7, "val": 0.15, "test": 0.15}, seed=3)
    first = splitter.split(table, "labelled_frames")
    table["partition"] = table.video_id.map(first)
    assert (table.groupby("session").partition.nunique() == 1).all()
    for cls in ("BIRD", "DRONE"):
        assert set(table[table.cls == cls].partition) == {"train", "val", "test"}
    again = GroupSplitter({"train": 0.7, "val": 0.15, "test": 0.15}, seed=3)
    assert again.split(table, "labelled_frames") == first


def test_too_few_sessions_is_an_error():
    table = _table().head(6)  # bird sessions 0 and 1 only
    table["session"] = assign_sessions(table, gap_minutes=30)
    splitter = GroupSplitter({"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)
    with pytest.raises(ValueError):
        splitter.split(table, "labelled_frames")
