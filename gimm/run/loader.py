from typing import Optional

from gimm.datasets.definition import Dataset


def dataset_loader(
        name: str,
        batch_size: int,
        num_workers: int,
        data_dir: str,
        split_config: Optional[list[int]] = None,
        bake: bool = False,
        bake_type: str = 'memory',
        bake_path: Optional[str] = None,
) -> Dataset:
    kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'data_dir': data_dir,
        'split_config': split_config,
        'bake': bake,
        'bake_type': bake_type,
        'bake_path': bake_path,
    }

    if name == "cifar10":
        from gimm.datasets.benchmark.cifar10 import DatasetCifar10
        return DatasetCifar10(**kwargs)
    elif name == "mnist":
        from gimm.datasets.benchmark.mnist import DatasetMNIST
        return DatasetMNIST(**kwargs)
    elif name == "celeba":
        from gimm.datasets.benchmark.celeba import DatasetCelebA
        return DatasetCelebA(**kwargs)
    elif name == "pistachio2":
        from gimm.datasets.debug.pistachio import DatasetPistachio1
        return DatasetPistachio1(**kwargs)
    elif name == "toucan":
        from gimm.datasets.debug.toucan import DatasetToucan
        return DatasetToucan(**kwargs)
    else:
        raise ValueError(f"Dataset {name} not found.")