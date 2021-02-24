from abc import ABCMeta, abstractmethod
import json
from pathgen.data.annotations import annotation
from pathlib import Path
from typing import Sequence

import pandas as pd
from pathgen.data import datasets

from pathgen.data.datasets import Dataset
from pathgen.data.datasets.registry import get_dataset
from pathgen.data.slides import Region


class PatchSet(metaclass=ABCMeta):
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "PatchSet":
        raise NotImplementedError

    def as_df(self) -> pd.DataFrame:
        rows = [
            (r.as_values(), slide_idx, dataset_name)
            for r, slide_idx, dataset_name in self
        ]
        frame = pd.DataFrame(
            rows, columns=["x", "y", "width", "height", "slide_idx", "dataset_name"]
        )
        return frame

     def export_patches(self, output_dir: Path) -> None:
        # not optimal!!!! order by dataset, the slide to reduce loading
        for region, slide_idx, dataset_name in self:
            dataset = get_dataset(dataset_name)
            with dataset.open_slide(slide_idx) as slide:
                image = slide.read_region(region)
                


class SimplePatchSet(PatchSet, Sequence):
    def __init__(
        self,
        dataset: Dataset,
        slide_idx: int,
        patch_size: int,
        level: int,
        patches_df: pd.DataFrame,
    ) -> None:
        self.dataset = dataset
        self.slide_idx = slide_idx
        self.patch_size = patch_size
        self.level = level
        self.patches_df = patches_df

    def __len__(self):
        return len(self.patches_df)

    def __getitem__(self, idx):
        row = self.patches_df.iloc[
            idx,
        ]
        region = Region.make(row.x, row.y, self.patch_size, self.level)
        return region, self.slide_idx, self.dataset.name

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
