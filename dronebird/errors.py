"""Wrong predictions with Grad-CAM heat maps, for the error analysis.

Author: Mariah Habak

Usage:
  python -m dronebird.errors results/resnet18_distance/seed0 --test-set real_small
  python -m dronebird.errors results/resnet18_distance/seed0 --test-set synthetic --size 16

Uses the validation set unless --final is given (then the test set).
Saves errors_<set>_<size>.png (crop + heat map) and a CSV with an empty
'group' column to fill in by hand after looking at the crops.
"""

import argparse
import csv
from pathlib import Path

import torch

from dronebird import plots
from dronebird.config import Config
from dronebird.data.crops import CropBank
from dronebird.evaluation import Evaluator
from dronebird.gradcam import GradCAM
from dronebird.models import build_model

NAMES = ("bird", "drone")


class ErrorAnalysis:
    def __init__(self, run_dir: Path, final: bool):
        self.__run_dir = Path(run_dir)
        config = Config.from_file(self.__run_dir / "config.json")
        data = Config.from_file(config.get("data_config", "configs/data.json")).section("data")
        crops = data["crops"]
        self.__split = "test" if final else "val"
        self.__bank = CropBank.load(Path(data["processed_dir"]) / f"{self.__split}.pt")
        self.__evaluator = Evaluator(data["evaluation"], crops["margin"], crops["canvas"])
        seed = int(self.__run_dir.name.replace("seed", ""))
        self.__model = build_model(config.section("model"), seed, crops["margin"],
                                   crops["canvas"])
        self.__model.load(self.__run_dir)

    def run(self, test_set: str, size: str, count: int) -> list:
        found = self.__evaluator.wrong_predictions(self.__model, self.__bank, test_set, size)
        scores, wrong, bank = found["scores"], found["wrong"], found["bank"]
        confidence = scores.max(dim=1).values
        ordered = wrong[confidence[wrong].argsort(descending=True)].tolist()
        seen_groups, picked = set(), []
        for i in ordered:  # most confident mistakes first, one per near-duplicate group
            if bank.group_ids[i] not in seen_groups:
                seen_groups.add(bank.group_ids[i])
                picked.append(i)
        wrong = torch.tensor(picked[:count], dtype=torch.long)
        network = getattr(self.__model, "network", None)
        cam = GradCAM(network, network.cam_layer()) if network is not None else None
        images, cams, captions, rows = [], [], [], []
        labels = bank.labels
        for rank, i in enumerate(wrong.tolist(), 1):
            image = found["images"][i]
            heat = cam(image)[0] if cam is not None else image[0] * 0
            images.append(image)
            cams.append(heat)
            captions.append(f"#{rank} true {NAMES[labels[i]]}, said "
                            f"{NAMES[scores[i].argmax()]} ({confidence[i]:.2f}), "
                            f"{bank.object_size[i]:.0f}px")
            rows.append({"rank": rank, "image": bank.image_ids[i],
                         "true": NAMES[labels[i]], "predicted": NAMES[scores[i].argmax()],
                         "score": round(float(confidence[i]), 3),
                         "object_size_px": round(float(bank.object_size[i]), 1),
                         "group": ""})
        stem = f"errors_{self.__split}_{test_set}_{size}"
        if images:
            plots.error_grid(torch.stack(images), cams, captions,
                             self.__run_dir / f"{stem}.png")
        with open(self.__run_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as h:
            writer = csv.DictWriter(h, fieldnames=["rank", "image", "true", "predicted",
                                                   "score", "object_size_px", "group"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"{len(wrong)} wrong predictions shown out of {len(found['wrong'])} "
              f"({len(labels)} crops) -> {self.__run_dir / stem}.png")
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--test-set", choices=["all", "synthetic", "real_small"],
                        default="real_small")
    parser.add_argument("--size", default="all",
                        help="'full' or 48/32/24/16 for synthetic; a bin name or 'all' for real_small")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    size = "full" if args.test_set == "all" else args.size
    if args.test_set == "synthetic" and size == "all":
        size = "full"
    ErrorAnalysis(args.run_dir, args.final).run(args.test_set, size, args.count)


if __name__ == "__main__":
    main()
