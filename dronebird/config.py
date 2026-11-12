"""Reading experiment configurations from JSON files.

Author: Mariah Habak
"""

import copy
import json
from pathlib import Path


class Config:
    """A read-only view of a JSON experiment configuration.

    Sections are reached with ``config.section("model")``; each call
    returns a copy, so no object can change the shared settings.
    """

    def __init__(self, values: dict, source: str = "<dict>"):
        self.__values = copy.deepcopy(values)
        self.__source = source

    @classmethod
    def from_file(cls, path) -> "Config":
        path = Path(path)
        with path.open(encoding="utf-8") as handle:
            return cls(json.load(handle), source=str(path))

    @property
    def source(self) -> str:
        return self.__source

    @property
    def name(self) -> str:
        return self.__values.get("name", Path(self.__source).stem)

    def section(self, key: str) -> dict:
        if key not in self.__values:
            raise KeyError(f"Config {self.__source} has no section '{key}'")
        return copy.deepcopy(self.__values[key])

    def get(self, key: str, default=None):
        return copy.deepcopy(self.__values.get(key, default))

    def as_dict(self) -> dict:
        return copy.deepcopy(self.__values)
