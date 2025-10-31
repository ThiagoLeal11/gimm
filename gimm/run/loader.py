from gimm.datasets.definition import Dataset


def dataset_loader(name: str, batch_size: int, num_workers: int, data_dir: str) -> Dataset:
    if name == "cifar10":
        from gimm.datasets.cifar10 import DatasetCifar10
        return DatasetCifar10(batch_size=batch_size, num_workers=num_workers, data_dir=data_dir)
    elif name == "mnist":
        from gimm.datasets.mnist import DatasetMNIST
        return DatasetMNIST(batch_size=batch_size, num_workers=num_workers, data_dir=data_dir)
    elif name == "toucan":
        from gimm.datasets.toucan import DatasetToucan
        return DatasetToucan(batch_size=batch_size, num_workers=num_workers, data_dir=data_dir)
    else:
        raise ValueError(f"Dataset {name} not found.")