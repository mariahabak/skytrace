"""One command that reruns the main experiment.

Author: Mariah Habak

  python -m dronebird.pipeline --smoke          # fake data, 2 epochs, CPU, ~3 min
  python -m dronebird.pipeline                  # real data: prepare + 4 models x 3 seeds
  python -m dronebird.pipeline --final          # ... then the ONE-TIME test evaluation
"""

import argparse
from pathlib import Path

from dronebird.config import Config
from dronebird.fake import make_fake_dataset
from dronebird.prepare import DataPreparer
from dronebird.run import Experiment
from dronebird.tables import SizeCurveReport

MODELS = ("majority", "small_cnn", "resnet18", "resnet18_distance")


class Pipeline:
    def __init__(self, config_dir: Path, results: str, final: bool):
        self.__configs = Path(config_dir)
        self.__results = results
        self.__final = final

    def run(self) -> None:
        data = Config.from_file(self.__configs / "data.json")
        DataPreparer(data).run()
        experiments = [Experiment(Config.from_file(self.__configs / f"{m}.json"), self.__results)
                       for m in MODELS]
        for experiment in experiments:
            for seed in experiment.seeds:
                experiment.fit(seed)
        self.__tables("val")
        if self.__final:
            for experiment in experiments:
                for seed in experiment.seeds:
                    experiment.test(seed)
            self.__tables("test")

    def __tables(self, split: str) -> None:
        print(SizeCurveReport(self.__results, self.__configs / "data.json", split).write())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="tiny fake dataset, CPU")
    parser.add_argument("--final", action="store_true", help="also evaluate on the test set")
    args = parser.parse_args()
    if args.smoke:
        root = Path(Config.from_file("configs/smoke/data.json").section("data")["dataset_root"])
        if not root.exists():
            make_fake_dataset(root)
        Pipeline(Path("configs/smoke"), "results_smoke", final=True).run()
    else:
        Pipeline(Path("configs"), "results", final=args.final).run()


if __name__ == "__main__":
    main()
