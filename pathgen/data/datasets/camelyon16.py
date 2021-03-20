from pathlib import Path
from typing import Dict

import pandas as pd

from pathgen.data.annotations import AnnotationSet
from pathgen.data.annotations.asapxml import load_annotations
from pathgen.data.datasets import Dataset
from pathgen.data.slides.openslide import Slide
from pathgen.data.slides import SlideBase
from pathgen.utils.paths import project_root


class Camelyon16(Dataset):
    def __init__(self, name: str, root: Path, paths: pd.DataFrame) -> None:
        super().__init__(name, root, paths)

    def load_annotations(self, file: Path) -> AnnotationSet:
        # if there is no annotation file the just pass and empty list
        annotations = load_annotations(file) if file else []
        labels_order = ["background", "tumor", "normal"]
        return AnnotationSet(annotations, self.labels, labels_order, "normal")

    @property
    def slide_cls(self) -> SlideBase:
        return Slide

    @property
    def labels(self) -> Dict[str, int]:
        return {"background": 0, "normal": 1, "tumor": 2}


def training():
    # set up the paths to the slides and annotations
    root = project_root() / "data" / "camelyon16" / "raw" / "training"
    annotations_dir = root / "lesion_annotations"
    tumor_slide_dir = root / "tumor"
    normal_slide_dir = root / "normal"

    # all paths are relative to the dataset 'root'
    annotation_paths = sorted(
        [p.relative_to(root) for p in annotations_dir.glob("*.xml")]
    )
    tumor_slide_paths = sorted(
        [p.relative_to(root) for p in tumor_slide_dir.glob("*.tif")]
    )
    normal_slide_paths = sorted(
        [p.relative_to(root) for p in normal_slide_dir.glob("*.tif")]
    )

    # turn them into a data frame and pad with empty annotation paths
    df = pd.DataFrame()
    df["slide"] = tumor_slide_paths + normal_slide_paths
    df["annotation"] = annotation_paths + ["" for _ in range(len(normal_slide_paths))]
    df["label"] = ["tumor"] * len(tumor_slide_paths) + ["normal"] * len(
        normal_slide_paths
    )
    df["tags"] = ""

    return Camelyon16("camelyon16.training", root, df)


def training_small():
    # set up the paths to the slides and annotations
    root = project_root() / "data" / "camelyon16" / "raw" / "training"
    annotations_dir = root / "lesion_annotations"
    tumor_slide_dir = root / "tumor"
    normal_slide_dir = root / "normal"

    # all paths are relative to the dataset 'root'
    annotation_paths = sorted(
        [p.relative_to(root) for p in annotations_dir.glob("*.xml")]
    )
    tumor_slide_paths = sorted(
        [p.relative_to(root) for p in tumor_slide_dir.glob("*.tif")]
    )
    normal_slide_paths = sorted(
        [p.relative_to(root) for p in normal_slide_dir.glob("*.tif")]
    )

    # turn them into a data frame and pad with empty annotation paths
    df = pd.DataFrame()
    df["slide"] = tumor_slide_paths + normal_slide_paths
    df["annotation"] = annotation_paths + ["" for _ in range(len(normal_slide_paths))]
    df["label"] = ["tumor"] * len(tumor_slide_paths) + ["normal"] * len(
        normal_slide_paths
    )
    df["tags"] = ""

    # df = df.sample(10)
    slides_for_small = [
        "normal/normal_014.tif",
        "normal/normal_038.tif",
        "normal/normal_100.tif",
        "tumor/tumor_024.tif",
        "tumor/tumor_038.tif",
        "tumor/tumor_054.tif",
        "tumor/tumor_063.tif",
        "tumor/tumor_065.tif",
        "tumor/tumor_076.tif",
        "tumor/tumor_089.tif",
    ]
    paths_for_small = [Path(s) for s in slides_for_small]
    df = df.loc[df["slide"].isin(paths_for_small)]
    df = df.reindex()

    return Camelyon16("camelyon16.training_small", root, df)


def testing():
    # set up the paths to the slides and annotations
    root = project_root() / "data" / "camelyon16" / "raw" / "testing"
    annotations_dir = root / "lesion_annotations"
    slide_dir = root / "images"

    # all paths are relative to the dataset 'root'
    slide_paths = sorted([p.relative_to(root) for p in slide_dir.glob("*.tif")])
    annotation_paths = sorted(
        [p.relative_to(root) for p in annotations_dir.glob("*.xml")]
    )

    # get the slide name
    slide_names = [p.stem for p in slide_paths]

    # search for slides with annotations, add the annotation path if it exists else add empty string
    slides_annotations_paths = []
    for name in slide_names:
        a_path = ""
        for anno_path in annotation_paths:
            if name in str(anno_path):
                a_path = anno_path
        slides_annotations_paths.append(a_path)

    # get the slide labels by reading the csv file
    csv_path = root / "reference.csv"
    label_csv_file = pd.read_csv(csv_path, header=None)
    slide_labels = label_csv_file.iloc[:, 1]

    # turn them into a data frame and pad with empty annotation paths
    df = pd.DataFrame()
    df["slide"] = slide_paths
    df["annotation"] = slides_annotations_paths
    df["label"] = slide_labels
    df["tags"] = ""

    return Camelyon16(root, df)
