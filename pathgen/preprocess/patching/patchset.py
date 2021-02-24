from abc import ABCMeta, abstractmethod
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from pathgen.data.datasets import Dataset
from pathgen.data.datasets.registry import get_dataset


class PatchSet(metaclass=ABCMeta):
    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "PatchSet":
        raise NotImplementedError


class SimplePatchSet(PatchSet, Sequence):
    def __init__(
        self, dataset: Dataset, patch_size: int, level: int, patches_df: pd.DataFrame,
    ) -> None:
        self.dataset = dataset
        self.patch_size = patch_size
        self.level = level
        self.patches_df = patches_df

    def __len__(self):
        return len(self.patches_df)

    def __getitem__(self, idx):
        return self.patches_df.iloc[
            idx,
        ]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.patches_df.to_csv(path / "frame.csv")
        data = {
            "type": type(self).__name__,
            "fields": {
                "dataset": self.dataset.name,
                "patch_size": self.patch_size,
                "level": self.level,
            },
        }
        with open(path / "fields.json", "w") as outfile:
            json.dump(data, outfile)

    @classmethod
    def load(cls, path: Path) -> "PatchSet":
        frame = pd.read_csv(path / "frame.csv")
        with open(path / "fields.json") as json_file:
            fields = json.load(json_file)["fields"]
            dataset = get_dataset(fields["dataset"])
            patch_size = fields["patch_size"]
            level = fields["level"]
            cls(dataset, patch_size, level, frame)
