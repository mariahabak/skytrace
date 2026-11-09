"""Train and evaluate one experiment over several seeds.

Authors: Maria Habak, [Partner name]

Usage:
  python -m skytrace.run configs/frame_single.json --stage fit
  python -m skytrace.run configs/frame_single.json --stage test --final

"fit" trains on the training set, selects on the validation set and
saves the model and its validation scores. "test" touches the test set
and is meant to run once, at the end, after every choice is frozen, so
it needs the --final flag.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from skytrace import plots
from skytrace.config import Config
from skytrace.data.clips import ClipSet
from skytrace.data.degrade import apply as degrade
from skytrace.evaluation import metrics
from skytrace.evaluation.uncertainty import RejectionRule, TemperatureScaler
from skytrace.models import build_model
from skytrace.seeding import set_seed


def to_jsonable(value):
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def write_json(path, value) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as h:
        json.dump(to_jsonable(value), h, indent=2)


class Experiment:
    """One JSON configuration, run for each seed."""

    def __init__(self, config: Config, results_root: str = "results"):
        self.__config = config
        self.__data_dir = Path(config.section("data")["processed_dir"])
        self.__model_settings = config.section("model")
        self.__evaluation = config.section("evaluation")
        self.__root = Path(results_root) / config.name

    def run_dir(self, seed: int) -> Path:
        return self.__root / f"seed{seed}"

    def __load(self, partition: str) -> ClipSet:
        return ClipSet.load(self.__data_dir / f"{partition}.pt")

    def fit(self, seed: int) -> dict:
        set_seed(seed)
        run_dir = self.run_dir(seed)
        train, val = self.__load("train"), self.__load("val")
        model = build_model(self.__model_settings, seed)
        print(f"[{self.__config.name} seed {seed}] training {model.name}")
        history = model.fit(train, val, run_dir)
        model.save(run_dir)
        val_scores = model.predict_scores(val)
        result = metrics.summary(val.labels.numpy(), val_scores)
        np.savez(run_dir / "val_scores.npz", scores=val_scores,
                 labels=val.labels.numpy())
        write_json(run_dir / "history.json", history)
        write_json(run_dir / "val_metrics.json", result)
        if history and "val_loss" in history[0]:
            plots.loss_curves(history, run_dir / "loss_curves.png")
        write_json(run_dir / "config.json", self.__config.as_dict())
        print(f"[{self.__config.name} seed {seed}] validation macro-F1 "
              f"{result['macro_f1']:.4f}")
        return result

    def test(self, seed: int) -> dict:
        set_seed(seed)
        run_dir = self.run_dir(seed)
        test = self.__load("test")
        model = build_model(self.__model_settings, seed)
        model.load(run_dir)

        saved = np.load(run_dir / "val_scores.npz")
        scaler = TemperatureScaler()
        scaler.fit(saved["labels"], saved["scores"])
        rule = RejectionRule(self.__evaluation["target_coverage"])
        rule.fit(scaler.transform(saved["scores"]))

        scores = model.predict_scores(test)
        labels = test.labels.numpy()
        calibrated = scaler.transform(scores)
        result = metrics.summary(labels, scores)
        result["temperature"] = scaler.temperature
        result["ece_after_scaling"] = metrics.expected_calibration_error(labels, calibrated)
        result["rejection"] = rule.report(labels, calibrated)
        result["area_error_coverage"] = metrics.area_under_error_coverage(labels, calibrated)
        result["by_object_size"] = self.__by_size(test, labels, scores)
        result["robustness"] = self.__robustness(model, test, labels)

        np.savez(run_dir / "test_scores.npz", scores=scores, calibrated=calibrated,
                 labels=labels, box_side=test.box_side.numpy(),
                 video_ids=np.array(test.video_ids), starts=np.array(test.starts))
        write_json(run_dir / "test_metrics.json", result)
        coverage, error = metrics.error_coverage(labels, calibrated)
        plots.error_coverage({model.name: (coverage, error)},
                             run_dir / "error_coverage.png")
        plots.reliability_diagram(calibrated.max(axis=1),
                                  calibrated.argmax(axis=1) == labels,
                                  run_dir / "reliability.png")
        print(f"[{self.__config.name} seed {seed}] TEST macro-F1 {result['macro_f1']:.4f}")
        return result

    def __by_size(self, clips: ClipSet, labels, scores) -> dict:
        sides = clips.box_side.numpy()
        out = {}
        edges = self.__evaluation["size_bins"]
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (sides >= lo) & (sides < hi)
            if mask.sum() and len(set(labels[mask])) == 2:
                out[f"{lo}-{hi}px"] = {
                    "clips": int(mask.sum()),
                    "macro_f1": metrics.macro_f1(labels[mask], scores[mask].argmax(1))}
        return out

    def __robustness(self, model, clips: ClipSet, labels) -> dict:
        out = {}
        for kind, severities in self.__evaluation["robustness"].items():
            out[kind] = {}
            for severity in severities:
                degraded = clips.with_clips(degrade(clips.clips, kind, severity))
                s = model.predict_scores(degraded)
                out[kind][str(severity)] = metrics.macro_f1(labels, s.argmax(axis=1))
        return out


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

    config = Config.from_file(args.config)
    seeds = args.seeds if args.seeds is not None else config.get("seeds", [0, 1, 2])
    experiment = Experiment(config, args.results)
    if args.stage == "test" and not args.final:
        parser.error("the test stage touches the test set: add --final")
    for seed in seeds:
        if args.stage == "fit":
            experiment.fit(seed)
        else:
            experiment.test(seed)


if __name__ == "__main__":
    main()
