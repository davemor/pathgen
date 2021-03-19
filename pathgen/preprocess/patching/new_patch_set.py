import json
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from pathgen.data.slides import SlideBase, Region
from pathgen.data.datasets import Dataset, get_dataset
from pathgen.utils.geometry import Size


class PatchSet:
    def __init__(
        self,
        df: pd.DataFrame,
        patch_size: int = None,
        level: int = None,
        slide_index: str = None,
        dataset_name: str = None,
    ) -> None:
        self.df = df
        self._patch_size = patch_size
        self._level = level
        self._slide_index = slide_index
        self._dataset_name = dataset_name
        self._dataset = get_dataset(dataset_name) if dataset_name else None

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.patches_df.to_csv(path / "frame.csv")
        fields = {f for f in self.__dict__ if f not in ["df", "dataset"]}
        data = {"type": type(self).__name__, "fields": fields}
        with open(path / "fields.json", "w") as outfile:
            json.dump(data, outfile)

    @classmethod
    def load(cls, path: Path) -> "PatchSet":
        df = pd.read_csv(path / "frame.csv")
        with open(path / "fields.json") as json_file:
            fields = json.load(json_file)["fields"]
            dataset = get_dataset(fields["dataset_name"])
            fields["df"] = df
            fields["dataset"] = dataset
            cls(**fields)

    def has_column(self, name: str) -> bool:
        return name in self.df.columns

    def export(self, output_dir: Path) -> None:
        def export_patch(region: Region, slide: SlideBase, label: str) -> None:
            image = slide.read_region(region)

            # ensure the output directory exists
            output_subdir = output_dir / label
            output_subdir.mkdir(parents=True, exist_ok=True)

            # write out the patch

        # order by dataset for efficent slide opening
        sort_columns = [c for c in ["dataset_name", "slide_idx"] if self.has_column(c)]
        if len(sort_columns) > 0:
            self.df = self.df.sort_values(sort_columns, ignore_index=True)

        # for each row in the dataframe output the image
        current_dataset_name = None
        current_slide_idx = None
        current_slide = None
        for idx, row in self.df.iterrows():
            patch = PatchDetails(self, row)
            if (
                current_dataset_name != patch.dataset_name
                or current_slide_idx != patch.slide_idx
            ):
                if current_slide:
                    current_slide.close()
                current_slide = patch.dataset.open_slide(patch.slide_idx)


class PatchDetails:
    def __init__(self, ps: PatchSet, row: pd.Series) -> None:
        self.ps = ps
        self.fields = ps.__dict__
        self.row = row

    def get(self, key: str) -> Any:
        return self.row["key"] if key in self.row else self.fields[key]

    @property
    def patch_size(self) -> int:
        return self.get("patch_size")

    @property
    def level(self) -> int:
        return self.get("level")

    @property
    def slide_idx(self) -> int:
        return self.get("slide_index")

    @property
    def dataset_name(self) -> str:
        return self.get("dataset_name")

    @property
    def dataset(self) -> Dataset:
        return get_dataset(self.dataset_name)

    @property
    def region(self) -> Region:
        return Region.make(self.row["x"], self.row["y"], self.patch_size, self.level)

    @property
    def label(self) -> str:
        label_idx = self.row["label"]
        return self.dataset.labels_by_index[label_idx]

    @property
    def slide_path(self) -> Path:
        path = self.dataset.get_slide_path(self.slide_idx)
        return path
