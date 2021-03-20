import json
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import numpy as np

from pathgen.data.slides import SlideBase, Region
from pathgen.data.datasets import Dataset, get_dataset


class PatchDetails:
    def __init__(self, ps: "PatchSet", row: pd.Series) -> None:
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

    def export(self, output_dir: Path) -> None:
        def sort_patches_by_slide():
            possible_columns = ["dataset_name", "slide_idx"]
            sort_columns = [c for c in possible_columns if c in self.df.columns]
            if len(sort_columns) > 0:
                self.df = self.df.sort_values(sort_columns, ignore_index=True)

        def make_patch_path(p: PatchDetails) -> Path:
            subdir = output_dir / p.label
            subdir.mkdir(parents=True, exist_ok=True)
            filename = f"{p.slide_path.stem}-{p.region.location.x}-{p.region.location.y}-{p.level}.png"
            return subdir / filename

        def save_patch(region: Region, slide: SlideBase, filepath: Path) -> None:
            image = slide.read_region(region)
            cv2.imwrite(str(filepath), np.array(image))

        # for each row in the dataframe output the image
        sort_patches_by_slide()
        dataset_name, slide_idx, slide = None, None, None
        for _, row in self.df.iterrows():
            p = PatchDetails(self, row)
            if dataset_name != p.dataset_name or slide_idx != p.slide_idx:
                if slide:
                    slide.close()
                slide = p.dataset.open_slide(p.slide_idx)
            filepath = make_patch_path(p)
            save_patch(p.region, slide, filepath)

    def summary(self) -> pd.DataFrame:
        groups = self.df.groupby("label")
        labels = groups.labels.groups.keys()
        counts = groups.size().to_frame().T.rename(columns=labels)
        columns = list(labels.values())
        summary = pd.DataFrame(columns=columns)
        for l in labels.values():
            if l in counts:
                summary[l] = counts[l]
            else:
                summary[l] = 0
        summary = summary.replace(np.nan, 0)
        return summary
