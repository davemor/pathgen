import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import pandas as pd
import numpy as np

from pathgen.data.slides import SlideBase, Region
from pathgen.data.datasets import Dataset, get_dataset
from pathgen.utils.convert import invert


class PatchDetails:
    def __init__(self, ps: "PatchSet", row: pd.Series) -> None:
        self.ps = ps
        self.fields = ps.__dict__
        self.row = row

    def get(self, key: str) -> Any:
        return self.row[key] if key in self.row else self.fields[f"_{key}"]

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

    # propeties
    @property
    def labels(self) -> Dict[str, int]:
        if self._dataset:
            return self._dataset.labels
        else:
            pass  # TODO: get this working with multiple datasets

    # serialisation
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path / "frame.csv", index=False)
        exclude = ["df", "_dataset"]
        fields = {k: v for k, v in self.__dict__.items() if k not in exclude}
        fields = {k[1:]: v for k, v in fields.items()}
        data = {"type": type(self).__name__, "fields": fields}
        with open(path / "fields.json", "w") as outfile:
            json.dump(data, outfile)

    @classmethod
    def load(cls, path: Path) -> "PatchSet":
        df = pd.read_csv(path / "frame.csv")
        with open(path / "fields.json") as json_file:
            fields = json.load(json_file)["fields"]
            fields["df"] = df
        return cls(**fields)

    # patch outputs
    def export(self, output_dir: Path) -> None:
        def sort_patches_by_slide():
            possible_columns = ["dataset_name", "slide_index"]
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
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(filepath), np.array(opencv_image))

        # for each row in the dataframe output the image
        sort_patches_by_slide()
        dataset_name, slide_idx, slide = None, None, None
        print("Exporting patches for: ", end="")
        for _, row in self.df.iterrows():
            p = PatchDetails(self, row)
            if dataset_name != p.dataset_name or slide_idx != p.slide_idx:
                dataset_name = p.dataset_name
                slide_idx = p.slide_idx
                if slide:
                    slide.close()
                slide = p.dataset.open_slide(slide_idx)
                slide.open()
                print(f"{slide_idx}", end=", ")
            filepath = make_patch_path(p)
            save_patch(p.region, slide, filepath)
        print("Complete.")

    def summary(self) -> pd.DataFrame:
        groups = self.df.groupby("label")
        frame = groups.size().to_frame().T
        frame = frame.rename(columns=invert(self.labels))
        for label in self.labels:
            if label not in frame.columns:
                frame[label] = 0
        frame = frame[self.labels.keys()]
        return frame


def combine(patchsets: List[PatchSet]) -> PatchSet:
    def to_frame(ps: PatchSet) -> pd.DataFrame:
        frame = ps.df.copy(deep=True)
        for attr in ["patch_size", "level", "slide_index", "dataset_name"]:
            value = getattr(ps, f"_{attr}", None)
            if attr not in frame.columns:
                frame[attr] = value
        return frame

    # create one big data frame with all the patch data in it
    frames = [to_frame(ps) for ps in patchsets]
    combined_df = pd.concat(frames, ignore_index=True)

    # optimise
    def is_unique(s):
        a = s.to_numpy()
        return (a[0] == a).all()

    cols = ["patch_size", "level", "slide_index", "dataset_name"]
    args = {}
    for col in cols:
        if is_unique(combined_df[col]):
            series = combined_df[col]
            args[col] = series[0]
            combined_df.pop(col)
    args["df"] = combined_df

    return PatchSet(**args)
