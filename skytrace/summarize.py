"""Mean ± standard deviation over seeds, for every experiment.

Authors: Maria Habak, [Partner name]

Usage:  python -m skytrace.summarize [--split val|test]
Writes results/summary_<split>.md and results/summary_<split>.csv.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def collect(results_root: Path, split: str) -> pd.DataFrame:
    rows = []
    for exp_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        for seed_dir in sorted(exp_dir.glob("seed*")):
            path = seed_dir / f"{split}_metrics.json"
            if not path.exists():
                continue
            m = json.loads(path.read_text())
            row = {"experiment": exp_dir.name, "seed": seed_dir.name,
                   "macro_f1": m["macro_f1"],
                   "bird_recall": m["per_class"]["bird"]["recall"],
                   "drone_recall": m["per_class"]["drone"]["recall"],
                   "ece": m["ece"]}
            if split == "test":
                row["ece_scaled"] = m["ece_after_scaling"]
                row["aurc"] = m["area_error_coverage"]
                row["coverage"] = m["rejection"]["coverage"]
                row["error_answered"] = m["rejection"]["error_when_answered"]
                for kind, values in m["robustness"].items():
                    for severity, f1 in values.items():
                        row[f"{kind}@{severity}"] = f1
            rows.append(row)
    return pd.DataFrame(rows)


def mean_std(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.drop(columns=["seed"])
    grouped = numeric.groupby("experiment")
    mean, std = grouped.mean(), grouped.std(ddof=1)
    out = mean.copy().astype(object)
    for column in mean.columns:
        out[column] = [f"{m:.3f} ± {s:.3f}" if not np.isnan(s) else f"{m:.3f}"
                       for m, s in zip(mean[column], std[column])]
    out.insert(0, "seeds", grouped.size())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    root = Path(args.results)
    frame = collect(root, args.split)
    if frame.empty:
        print("No results found.")
        return
    table = mean_std(frame)
    frame.to_csv(root / f"summary_{args.split}_per_seed.csv", index=False)
    table.to_csv(root / f"summary_{args.split}.csv")
    (root / f"summary_{args.split}.md").write_text(table.to_markdown())
    print(table.to_markdown())


if __name__ == "__main__":
    main()
