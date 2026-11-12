"""Size-curve tables and figures: mean ± std over seeds.

Author: Mariah Habak

Usage:  python -m dronebird.tables --split val      (or --split test)

All results are kept in dictionaries:
    results[(model, test_set, size, seed)] = {"accuracy": .., "macro_f1": ..}
    table[(model, test_set, size)]        = (mean, std)       per metric
Writes results/size_curves_<split>.md and figures in results/figures/.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dronebird import plots
from dronebird.config import Config

METRICS = ("accuracy", "macro_f1")
DEFAULT_ORDER = ["majority", "small_cnn", "resnet18", "resnet18_distance"]


class ResultTable:
    def __init__(self, results: dict):
        self.__results = results

    @classmethod
    def collect(cls, root: Path, split: str) -> "ResultTable":
        results = {}
        for path in sorted(root.glob(f"*/seed*/{split}_results.json")):
            for r in json.loads(path.read_text()):
                results[(r["model"], r["test_set"], r["size"], r["seed"])] = {
                    "accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "n": r["n"]}
        return cls(results)

    @property
    def models(self) -> list:
        found = {m for m, _, _, _ in self.__results}
        ordered = [m for m in DEFAULT_ORDER if m in found]
        return ordered + sorted(found - set(ordered))

    def aggregate(self, metric: str) -> dict:
        values = {}
        for (model, test_set, size, _seed), r in self.__results.items():
            values.setdefault((model, test_set, size), []).append(r[metric])
        out = {}
        for key, v in values.items():
            a = np.asarray(v, dtype=float)
            std = float(np.nanstd(a, ddof=1)) if np.isfinite(a).sum() > 1 else float("nan")
            out[key] = (float(np.nanmean(a)) if np.isfinite(a).any() else float("nan"), std)
        return out

    def counts(self) -> dict:
        return {(m, t, s): r["n"] for (m, t, s, _), r in self.__results.items()}

    def seeds(self, model: str) -> int:
        return len({seed for (m, _, _, seed) in self.__results if m == model})

    def markdown(self, metric: str, test_set: str, columns: list) -> str:
        table = self.aggregate(metric)
        n = self.counts()
        first = self.models[0] if self.models else None
        header = "| model | seeds | " + " | ".join(f"{c} (n={n.get((first, test_set, c), 0)})"
                                                for c in columns) + " |"
        lines = [header, "|" + "---|" * (len(columns) + 2)]
        for model in self.models:
            cells = []
            for c in columns:
                mean, std = table.get((model, test_set, c), (float("nan"), float("nan")))
                cells.append("—" if np.isnan(mean) else
                             f"{100 * mean:.1f} ± {100 * std:.1f}" if not np.isnan(std)
                             else f"{100 * mean:.1f}")
            lines.append(f"| {model} | {self.seeds(model)} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def hypothesis(self, plain: str, augmented: str, small: list, full: str = "full") -> str:
        """Augmented minus plain, in accuracy points, at small sizes and at full size."""
        acc = self.aggregate("accuracy")
        lines = [f"Accuracy difference {augmented} − {plain} (points, mean over seeds):"]
        for test_set, sizes in (("synthetic", small + [full]), ("real_small", small + ["all"])):
            for size in sizes:
                a = acc.get((augmented, test_set, size), (np.nan,))[0]
                b = acc.get((plain, test_set, size), (np.nan,))[0]
                unit = "" if size in ("full", "all") else " px"
                lines.append(f"  {test_set:10s} {size:>4s}{unit}: {100 * (a - b):+.1f}")
        return "\n".join(lines)


class SizeCurveReport:
    """Writes the markdown tables and the size-curve figures for one split."""

    def __init__(self, results_root, data_config, split: str):
        self.__root = Path(results_root)
        self.__split = split
        self.__ev = Config.from_file(data_config).section("data")["evaluation"]

    def write(self) -> str:
        ev, split, root = self.__ev, self.__split, self.__root
        synthetic_cols = ["full"] + [str(s) for s in ev["synthetic_sizes"]]
        real_cols = list(ev["real_size_bins"]) + ["all"]
        bins = ", ".join(f"{k}: {v[0]}–{v[1]} px" for k, v in ev["real_size_bins"].items())
        table = ResultTable.collect(root, split)
        if not table.models:
            return "No results found."
        parts = [f"# Size curves ({split}), mean ± std over seeds, in %\n"]
        for metric in METRICS:
            parts.append(f"## Synthetic small targets — {metric}\n"
                         f"Objects ≥ {ev['synthetic_min_size']} px, shrunk with the test recipe.\n")
            parts.append(table.markdown(metric, "synthetic", synthetic_cols) + "\n")
            parts.append(f"## Real small targets — {metric}\n"
                         f"Objects < {ev['real_max_size']} px in the frame, unchanged. "
                         f"Bins: {bins}.\n")
            parts.append(table.markdown(metric, "real_small", real_cols) + "\n")
            parts.append(f"## All {split} crops — {metric}\n")
            parts.append(table.markdown(metric, "all", ["full"]) + "\n")
        if {"resnet18", "resnet18_distance"} <= set(table.models):
            small = [s for s in ("24", "16") if s in synthetic_cols]
            parts.append("## Hypothesis check\n```\n"
                         + table.hypothesis("resnet18", "resnet18_distance", small) + "\n```\n")
        text = "\n".join(parts)
        (root / f"size_curves_{split}.md").write_text(text, encoding="utf-8")
        for metric in METRICS:
            agg = table.aggregate(metric)
            plots.size_curves(agg, table.models, synthetic_cols, "synthetic", metric,
                              root / "figures" / f"{split}_synthetic_{metric}.png",
                              f"Synthetic small targets ({split})")
            plots.size_curves(agg, table.models, real_cols, "real_small", metric,
                              root / "figures" / f"{split}_real_small_{metric}.png",
                              f"Real small targets ({split})")
        return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--results", default="results")
    parser.add_argument("--data-config", default="configs/data.json")
    args = parser.parse_args()
    print(SizeCurveReport(args.results, args.data_config, args.split).write())


if __name__ == "__main__":
    main()
