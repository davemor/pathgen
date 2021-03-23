from pathgen.data.datasets.dataset import Dataset
from typing import List, Sequence
from pathlib import Path

import pandas as pd

from pathgen.preprocess.patching.patchset import PatchSet


class SlidesIndex(Sequence):
    def __init__(self, patches: List[PatchSet]) -> None:
        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]

    def summary(self) -> pd.DataFrame:
        summaries = [s.summary() for s in self.patches]
        rtn = pd.concat(summaries)
        rtn = rtn.reset_index()
        rtn = rtn.drop("index", axis=1)
        return rtn

    def save(self, output_dir: Path) -> None:
        for idx, patchset in enumerate(self.patches):
            patchset.save(output_dir / f"{idx}")

    @classmethod
    def load(cls, input_dir: Path) -> "SlidesIndex":
        subdirs = [x for x in input_dir.iterdir() if x.is_dir()]
        subdirs = sorted(subdirs)  # might not be required
        patches = [PatchSet.load(subdir) for subdir in subdirs]
        return cls(patches)

    def select(self, indices: List[int]) -> "SlidesIndex":
        patchsets = [self[i] for i in indices]
        return SlidesIndex(patchsets)

