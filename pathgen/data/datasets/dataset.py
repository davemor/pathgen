from abc import ABCMeta, abstractmethod
from collections import Sequence
from pathlib import Path
from typing import Dict

import pandas as pd

from pathgen.data.slides.slide import SlideBase
from pathgen.data.annotations.annotation import AnnotationSet
from pathgen.utils.paths import project_root


class Dataset(Sequence, metaclass=ABCMeta):
    """ A data set is an object that represents a set of slides and their annotations.

    It can be used to load and iterate over a set of slides and their annotations.
    This is an abstract base class where classes that represent specific data sets
    should overload the load_annotations and slide_cls methods. See their descriptions
    for details.
    It implements the Sequence protocol so that it can be iterated over.

    Args:
        Sequence ([type]): [description]
        metaclass ([type], optional): [description]. Defaults to ABCMeta.
    """

    def __init__(self, name: str, root: Path, paths: pd.DataFrame) -> None:
        # process the paths_df (has two columns 'slide', 'annotation', 'label', 'tags')
        # root is stored relative to project_root
        self.name = name
        self.root = root.relative_to(project_root())
        self.paths = paths

    @abstractmethod
    def load_annotations(file: Path) -> AnnotationSet:
        raise NotImplementedError

    @property
    @abstractmethod
    def slide_cls(self) -> SlideBase:
        raise NotImplementedError

    @property
    def labels(self) -> Dict[str, int]:
        raise NotImplementedError

    @property
    def labels_by_index(self) -> Dict[str, int]:
        return {v: k for k, v in self.labels.items()}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        row = self.paths.iloc[idx]
        slide_path = self.to_abs_path(row["slide"])
        annot_path = (
            self.to_abs_path(row["annotation"]) if row["annotation"] != "" else ""
        )
        return slide_path, annot_path, row["label"], row["tags"]

    def to_abs_path(self, path: Path) -> Path:
        return project_root() / self.root / path

    def to_rel_path(self, path: Path) -> Path:
        return path.relative_to(project_root() / self.root)

    def load_annotations_for_slide(self, idx: int) -> AnnotationSet:
        row = self.paths.iloc[idx]
        if row["annotation"] == "":
            return []
        else:
            path = self.to_abs_path(row["annotation"])
            return self.load_annotations(path)

    def open_slide(self, idx: int):
        row = self.paths.iloc[idx]
        slide_path = self.to_abs_path(row["slide"])
        return self.slide_cls(slide_path)

    def get_slide_path(self, idx: int):
        row = self.paths.iloc[idx]
        slide_path = self.to_abs_path(row["slide"])
        return slide_path

    # def split_by_slide_label(self, label: str) -> List[Dataset]:
    #    self.paths.groupby("label")

