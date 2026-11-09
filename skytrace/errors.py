"""Collect wrong test predictions for the error analysis (C7).

Authors: Maria Habak, [Partner name]

Usage:  python -m skytrace.errors results/temporal_gru/seed0 --count 10
Saves errors.png (each wrong clip with its true class, prediction and
score) and errors.csv with an empty 'group' column for you to fill in
by hand after looking at the clips (for example: tiny object, bird
gliding, cloud edge, drone turning...).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from skytrace.config import Config  # noqa: E402
from skytrace.data.clips import ClipSet  # noqa: E402

NAMES = ("bird", "drone")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    config = Config.from_file(run_dir / "config.json")
    test = ClipSet.load(Path(config.section("data")["processed_dir"]) / "test.pt")
    saved = np.load(run_dir / "test_scores.npz")
    scores, labels = saved["calibrated"], saved["labels"]
    wrong = np.flatnonzero(scores.argmax(1) != labels)
    # Most confident mistakes first: these are the most informative.
    wrong = wrong[np.argsort(-scores[wrong].max(1))][: args.count]

    clips = test.clips.numpy()
    length = clips.shape[1]
    fig, axes = plt.subplots(len(wrong), length,
                             figsize=(length * 0.9, max(1, len(wrong)) * 1.05),
                             squeeze=False)
    rows = []
    for r, i in enumerate(wrong):
        for t in range(length):
            axes[r, t].imshow(clips[i, t], cmap="gray", vmin=0, vmax=255)
            axes[r, t].set_xticks([])
            axes[r, t].set_yticks([])
        axes[r, 0].set_ylabel(f"#{r + 1}", fontsize=8)
        axes[r, -1].set_title(f"true {NAMES[labels[i]]}, said {NAMES[scores[i].argmax()]}"
                              f" ({scores[i].max():.2f})", fontsize=6, loc="right")
        rows.append({"rank": r + 1, "video_id": test.video_ids[i],
                     "start_frame": test.starts[i], "true": NAMES[labels[i]],
                     "predicted": NAMES[scores[i].argmax()],
                     "score": round(float(scores[i].max()), 3),
                     "box_side_px": round(float(saved["box_side"][i]), 1),
                     "group": ""})
    fig.tight_layout()
    fig.savefig(run_dir / "errors.png", dpi=150)
    pd.DataFrame(rows).to_csv(run_dir / "errors.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
