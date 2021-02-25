from abc import ABCMeta, abstractmethod
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd

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

    @abstractmethod
    def export_patches(self, output_dir: Path) -> None:
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
        self.slide_path = dataset.get_slide_path(slide_idx)
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

    def export_patches(self, output_dir: Path) -> None:
        with self.dataset.open_slide(self.slide_idx) as slide:
            print(f"Writing patches for slide {self.slide_idx}")
            for row in self.patches_df.itertuples():
                # read the patch image from the slide
                region = Region.patch(row.x, row.y, self.patch_size, self.level)
                image = slide.read_region(region)

                # get the patch label as a string
                labels = {v: k for k, v in self.dataset.labels.items()}
                label = labels[row.label]

                # ensure the output directory exists
                output_subdir = output_dir / label
                output_subdir.mkdir(parents=True, exist_ok=True)

                # write out the slide
                rel_slide_path = self.dataset.to_rel_path(self.slide_path)
                slide_name_str = str(rel_slide_path)[:-4].replace("/", "-")
                patch_filename = slide_name_str + f"-{row.x}-{row.y}.png"
                image_path = output_dir / label / patch_filename
                cv2.imwrite(str(image_path), np.array(image))

    def summary(self) -> pd.DataFrame:
        """Gives a summary of number of patches for each class as a dataframe.
        Returns:
            pd.DataFrame: A summary dataframe defining number of patches for each class
        """
        by_label = self.patches_df.groupby("label").size()
        labels = {v: k for k, v in self.dataset.labels.items()}
        count_df = by_label.to_frame().T.rename(columns=labels)
        columns = list(labels.values())
        summary = pd.DataFrame(columns=columns)
        for l in labels.values():
            if l in count_df:
                summary[l] = count_df[l]
            else:
                summary[l] = 0
        summary = summary.replace(np.nan, 0)  # if there are no patches for some classes
        return summary

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
