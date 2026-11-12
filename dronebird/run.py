"""Train and evaluate one experiment over several seeds.

Author: Mariah Habak

Usage:
  python -m dronebird.run configs/resnet18.json --stage fit
  python -m dronebird.run configs/resnet18.json --stage test --final

"fit" trains on train, keeps the best epoch on validation, and writes
the validation size curves. "test" touches the test set and is meant to
run once, at the end, after every choice is frozen: it needs --final.
"""

import argparse
import json
from pathlib import Path

from dronebird import plots
from dronebird.config import Config
from dronebird.data.crops import CropBank
from dronebird.evaluation import Evaluator, to_records
from dronebird.models import build_model
from dronebird.seeding import set_seed


def write_json(path, value) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as h:
        json.dump(value, h, indent=2)


class Experiment:
    """One JSON configuration, run for each seed."""

    def __init__(self, config: Config, results_root: str = "results"):
        self.__config = config
        data = Config.from_file(config.get("data_config", "configs/data.json"))
        self.__data = data.section("data")
        self.__crops = self.__data["crops"]
        self.__processed = Path(self.__data["processed_dir"])
        self.__model_settings = config.section("model")
        self.__evaluator = Evaluator(self.__data["evaluation"], self.__crops["margin"],
                                     self.__crops["canvas"])
        self.__root = Path(results_root) / config.name

    @property
    def seeds(self) -> list:
        return self.__config.get("seeds", [0, 1, 2])

    def run_dir(self, seed: int) -> Path:
        return self.__root / f"seed{seed}"

    def __bank(self, partition: str) -> CropBank:
        return CropBank.load(self.__processed / f"{partition}.pt")

    def __model(self, seed: int):
        return build_model(self.__model_settings, seed, self.__crops["margin"],
                           self.__crops["canvas"])

    def fit(self, seed: int) -> dict:
        set_seed(seed)
        run_dir = self.run_dir(seed)
        train, val = self.__bank("train"), self.__bank("val")
        model = self.__model(seed)
        print(f"[{self.__config.name} seed {seed}] training {model.name} "
              f"on {len(train)} crops", flush=True)
        history = model.fit(train, val, run_dir)
        model.save(run_dir)
        results = self.__evaluator.size_curve(model, val, "val")
        write_json(run_dir / "history.json", history)
        write_json(run_dir / "val_results.json",
                   to_records(results, self.__config.name, seed, "val"))
        write_json(run_dir / "config.json", self.__config.as_dict())
        plots.loss_curves(history, run_dir / "loss_curves.png")
        print(f"[{self.__config.name} seed {seed}] validation accuracy "
              f"{results[('all', 'full')]['accuracy']:.4f}", flush=True)
        return results

    def test(self, seed: int) -> dict:
        set_seed(seed)
        run_dir = self.run_dir(seed)
        model = self.__model(seed)
        model.load(run_dir)
        results = self.__evaluator.size_curve(model, self.__bank("test"), "test")
        write_json(run_dir / "test_results.json",
                   to_records(results, self.__config.name, seed, "test"))
        print(f"[{self.__config.name} seed {seed}] TEST accuracy "
              f"{results[('all', 'full')]['accuracy']:.4f}", flush=True)
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--stage", choices=["fit", "test"], default="fit")
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--final", action="store_true",
                        help="confirm that this is the one-time test evaluation")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    if args.stage == "test" and not args.final:
        parser.error("the test stage touches the test set: add --final")
    experiment = Experiment(Config.from_file(args.config), args.results)
    for seed in (args.seeds if args.seeds is not None else experiment.seeds):
        getattr(experiment, args.stage)(seed)


if __name__ == "__main__":
    main()
