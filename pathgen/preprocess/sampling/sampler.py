from copy import copy
from typing import Callable

import numpy as np
import pandas as pd

from pathgen.preprocess.patching import PatchSet

SamplingPolicy = Callable[[pd.DataFrame, int], pd.DataFrame]


def weighted_random(class_df: pd.DataFrame, sum_totals: int) -> pd.DataFrame:
    class_df = class_df.assign(
        freq=class_df.groupby("slide_index")["slide_index"].transform("count").tolist()
    )
    class_df = class_df.assign(weights=np.divide(1, class_df.freq))
    class_sample = class_df.sample(
        n=sum_totals, axis=0, replace=False, weights=class_df.weights
    )
    return class_sample


def sample(
    ps: PatchSet,
    num_samples_per_class: int,
    floor_samples: int = 1000,
    sampling_policy: SamplingPolicy = weighted_random,
) -> PatchSet:
    frame = ps.df
    labels = np.unique(frame.label)
    sum_totals = [np.sum(frame.label == label) for label in labels]

    # find the count for the class with the smallest number of samples
    n_patches = min(sum_totals)

    # limit the count to the number of samples that we want
    n_patches = min(n_patches, num_samples_per_class)

    # make sure we are above the floor
    n_patches = max(n_patches, floor_samples)
    sum_totals = np.minimum(sum_totals, n_patches)

    sampled_patches = pd.DataFrame(columns=frame.columns)
    for idx, label in enumerate(labels):
        class_df = frame[frame.label == label]
        class_sample = sampling_policy(class_df, sum_totals[idx])
        sampled_patches = pd.concat([sampled_patches, class_sample], axis=0)

    # filter columns
    possible_cols = [
        "x",
        "y",
        "label",
        "slide_index",
        "patch_size",
        "level",
        "dataset_name",
    ]
    required_cols = [col for col in sampled_patches.columns if col in possible_cols]
    sampled_patches = sampled_patches[required_cols]

    sampled_patchset = copy(ps)
    sampled_patchset.df = sampled_patches
    return sampled_patchset
