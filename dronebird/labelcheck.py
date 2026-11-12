"""Checking 200 labels by hand (promised in the proposal).

Author: Mariah Habak

Step 1:  python -m dronebird.labelcheck sheet
         -> data/processed/labelcheck/sheet_1.png ... and labelcheck.csv
Step 2:  open the sheets, and in labelcheck.csv write "no" in the column
         'label_correct' for every crop whose label is wrong ("yes" otherwise).
Step 3:  python -m dronebird.labelcheck score
         -> prints the error rate with a 95% confidence interval.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from dronebird.config import Config
from dronebird.data.crops import CropBank

NAMES = ("bird", "drone")
PER_SHEET = 50


class LabelCheck:
    def __init__(self, config: Config, count: int = 200, seed: int = 0):
        data = config.section("data")
        self.__bank = CropBank.load(Path(data["processed_dir"]) / "train.pt")
        self.__dir = Path(data["processed_dir"]) / "labelcheck"
        self.__count = count
        self.__seed = seed

    def sheet(self) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(self.__seed)
        n = min(self.__count, len(self.__bank))
        chosen = sorted(rng.choice(len(self.__bank), n, replace=False).tolist())
        self.__dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for sheet_no, start in enumerate(range(0, n, PER_SHEET), 1):
            part = chosen[start:start + PER_SHEET]
            fig, axes = plt.subplots(5, 10, figsize=(13, 7.5))
            for ax in axes.flat:
                ax.axis("off")
            for k, i in enumerate(part):
                ax = axes.flat[k]
                ax.imshow(self.__bank.images[i].permute(1, 2, 0).numpy())
                ax.set_title(f"{start + k + 1}: {NAMES[self.__bank.labels[i]]}", fontsize=7)
                rows.append({"number": start + k + 1, "image": self.__bank.image_ids[i],
                             "label": NAMES[self.__bank.labels[i]],
                             "object_size_px": round(float(self.__bank.object_size[i]), 1),
                             "label_correct": ""})
            fig.tight_layout()
            fig.savefig(self.__dir / f"sheet_{sheet_no}.png", dpi=110)
            plt.close(fig)
        path = self.__dir / "labelcheck.csv"
        with open(path, "w", newline="", encoding="utf-8") as h:
            writer = csv.DictWriter(h, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def score(self) -> dict:
        with open(self.__dir / "labelcheck.csv", encoding="utf-8") as h:
            answers = [r["label_correct"].strip().lower() for r in csv.DictReader(h)]
        checked = [a for a in answers if a in ("yes", "no")]
        wrong = checked.count("no")
        n = len(checked)
        rate = wrong / n if n else float("nan")
        # Wilson 95% interval for a proportion
        z = 1.96
        centre = (rate + z * z / (2 * n)) / (1 + z * z / n) if n else float("nan")
        half = (z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
                if n else float("nan"))
        return {"checked": n, "wrong": wrong, "error_rate": rate,
                "ci95": [max(0.0, centre - half), min(1.0, centre + half)],
                "unanswered": len(answers) - n}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["sheet", "score"])
    parser.add_argument("--config", default="configs/data.json")
    args = parser.parse_args()
    check = LabelCheck(Config.from_file(args.config))
    print(check.sheet() if args.step == "sheet" else check.score())


if __name__ == "__main__":
    main()
