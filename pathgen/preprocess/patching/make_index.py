from pathlib import Path
from typing import List

from pathgen.data.datasets import Dataset
from pathgen.preprocess.tissue_detector.tissue_detector import TissueDetector
from pathgen.preprocess.patching.patch_finder import PatchFinder
from pathgen.preprocess.patching.patchset import SimplePatchSet
from pathgen.preprocess.patching.slide_index import SlidesIndex


def index_slide(
    slide_idx: int,
    dataset: Dataset,
    tissue_detector: TissueDetector,
    patch_finder: PatchFinder,
):
    slide_path, annotation_path, _, _ = dataset[slide_idx]
    with dataset.slide_cls(slide_path) as slide:
        print(f"indexing {slide_path.name}")  # TODO: Add proper logging!
        annotations = dataset.load_annotations(annotation_path)
        labels_shape = slide.dimensions[patch_finder.labels_level].as_shape()
        scale_factor = 2 ** patch_finder.labels_level
        labels_image = annotations.render(labels_shape, scale_factor)
        tissue_mask = tissue_detector(slide.get_thumbnail(patch_finder.labels_level))
        labels_image[~tissue_mask] = 0
        df, level, size = patch_finder(
            labels_image, slide.dimensions[patch_finder.patch_level]
        )
        patchset = SimplePatchSet(dataset, slide_idx, size, level, df)
        return patchset


def make_index(
    dataset: Dataset, tissue_detector: TissueDetector, patch_finder: PatchFinder
) -> SlidesIndex:
    patchsets = [
        index_slide(idx, dataset, tissue_detector, patch_finder)
        for idx in range(len(dataset))
    ]
    return SlidesIndex(dataset, patchsets)
