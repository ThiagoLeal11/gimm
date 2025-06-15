from abc import ABC, abstractmethod
from typing import Literal, Optional, Sequence

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2


class Dataset(ABC):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 4,
        data_dir: str = ".",
        *,
        seed: int = 42,
        shuffle: bool = True,
        # Reshuffle the dataset at each epoch using the seed + 1
        incremental_shuffle: bool = True,
        prefetch_factor: int = 1,
        pin_memory_device: torch.device = 'cpu',
        persistent_workers: bool = True,
        # List of transformations to apply to the dataset, defaults to [ToTensor(), Normalize(0.5)]
        transformations: Optional[list[v2.Compose]] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.incremental_shuffle = incremental_shuffle
        self.pin_memory_device: str = str(pin_memory_device) if str(pin_memory_device).lower() != 'cpu' else ""
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

        torch.manual_seed(self.seed)

        self.dims: Optional[Sequence[int]] = None
        self.num_classes: Optional[int] = None
        self.transformations = transformations
        self.definitions()

        assert self.dims is not None, "Dataset dimensions must be defined in definitions() method."
        assert self.num_classes is not None, "Number of classes must be defined in definitions() method."


        if self.transformations is None:
            t = [
                v2.Normalize((0.5,) * self.dims[0], (0.5,) * self.dims[0]),
            ]
            self.transformations = v2.Compose(t)

        self.prepare_data()
        self.dataset_train, self.dataset_val, self.dataset_test = None, None, None

    @abstractmethod
    def definitions(self):
        pass

    def prepare_data(self):
        pass

    @abstractmethod
    def setup(self, stage: Literal["train", "test"]):
        pass

    def train_dataloader(self):
        if self.dataset_train is None:
            self.setup("train")

        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=bool(self.pin_memory_device),
            pin_memory_device=self.pin_memory_device,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        if self.dataset_val is None:
            self.setup("train")

        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=bool(self.pin_memory_device),
            pin_memory_device=self.pin_memory_device,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        if self.dataset_test is None:
            self.setup("test")

        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=bool(self.pin_memory_device),
            pin_memory_device=self.pin_memory_device,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )
