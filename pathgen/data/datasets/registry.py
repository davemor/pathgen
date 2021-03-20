from pathgen.data.datasets import Dataset, camelyon16

datasets = {}


def get_dataset(name: str) -> Dataset:
    # print(f"calling get_dataset({name})")
    if name in datasets:
        return datasets[name]
    else:
        constructor = eval(name)
        dataset = constructor()
        datasets[name] = dataset
        return dataset
