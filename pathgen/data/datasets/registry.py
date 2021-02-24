from pathgen.data.datasets import Dataset

import pathgen.data.datasets.camelyon16 as camelyon16

datasets = {}


def get_dataset(name: str) -> Dataset:
    if name in datasets:
        return datasets[name]
    else:
        constructor = eval(name)
        dataset = constructor()
        datasets[name] = dataset
        return dataset
