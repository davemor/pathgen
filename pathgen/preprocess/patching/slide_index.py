from pathgen.data.datasets.dataset import Dataset
from typing import List, Sequence
from pathlib import Path

from pandas import pd

from pathgen.preprocess.patching.patchset import SimplePatchSet


class SlidesIndex(Sequence):
    def __init__(self, dataset: Dataset, patches: List[SimplePatchSet]) -> None:
        self.dataset = dataset
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
            patchset.save(output_dir / f"{idx:02}")

    @classmethod
    def load(cls, dataset: Dataset, input_dir: Path) -> "SlidesIndex":
        subdirs = [x for x in input_dir.iterdir() if x.is_dir()]
        subdirs = sorted(subdirs)  # might not be required
        patches = [SimplePatchSet.load(subdir) for subdir in subdirs]
        cls(dataset, patches)
