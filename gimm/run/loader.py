import torch

from gimm.datasets.definition import Dataset


def dataset_loader(name: str, batch_size: int, num_workers: int, device: torch.device, data_dir: str) -> Dataset:
    if name == "cifar10":
        from gimm.datasets.cifar10 import DatasetCifar10
        return DatasetCifar10(batch_size=batch_size, num_workers=num_workers, pin_memory_device=device, data_dir=data_dir)
    elif name == "mnist":
        from gimm.datasets.mnist import DatasetMNIST
        return DatasetMNIST(batch_size=batch_size, num_workers=num_workers, pin_memory_device=device, data_dir=data_dir)
    else:
        raise ValueError(f"Dataset {name} not found.")