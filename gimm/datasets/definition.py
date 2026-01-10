from abc import ABC, abstractmethod
from typing import Literal, Optional, Sequence, Callable

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
import logging



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
        transformations: Optional[Sequence[Callable]] = None,

        split_config: Optional[list[int]] = None,
    ):
        super().__init__()
        self.split: Optional[Sequence[int]] = None
        self.split_config = split_config

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

        self.dims: Sequence[int] | None = None
        self.num_classes: Optional[int] = None
        self.transformations = transformations
        self.definitions()

        assert self.dims is not None, "Dataset dimensions must be defined in definitions() method."
        assert self.num_classes is not None, "Number of classes must be defined in definitions() method."
        assert self.split is not None, "Dataset split must be defined in definitions() method."
        assert len(self.split) == 3, "Dataset split must be a list of three integers: [train_size, val_size, test_size]"

        if self.split_config:
            assert len(self.split_config) == 3, "split_config must be a list of three integers: [train_size, val_size, test_size]"
            self.split = self.split_config

        if self.transformations is None:
            t = [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.5,) * self.dims[0], (0.5,) * self.dims[0]),
            ]
            self.transformations = v2.Compose(t)

        self.prepare_data()
        self.dataset_train, self.dataset_val, self.dataset_test = None, None, None
        self._validation_applied = False
        self._loaders = {}

    def _split_train_val(self, full_train_dataset):
        split = self.get_splits()  # format: [train_size, val_size, test_size]
        train_size, val_size = split[:2]
        full_len = len(full_train_dataset)

        if train_size == -1:
            train_size = full_len - val_size
        elif val_size == -1:
            val_size = full_len - train_size

        if train_size + val_size != full_len:
             raise ValueError(f"Split sizes {train_size} + {val_size} do not match dataset length {full_len}")

        datasets =  random_split(
            full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(self.seed)
        )

        self._verify_split_sizes()
        return datasets

    def _verify_split_sizes(self):
        if self.split is None:
            return

        expected_train, expected_val, expected_test = self.split
        if self.dataset_train and len(self.dataset_train) != expected_train:
            logging.warning(f"Train dataset size mismatch: Expected {expected_train}, Got {len(self.dataset_train)}")
        if self.dataset_val and len(self.dataset_val) != expected_val:
            logging.warning(f"Validation dataset size mismatch: Expected {expected_val}, Got {len(self.dataset_val)}")
        if self.dataset_test and len(self.dataset_test) != expected_test:
            logging.warning(f"Test dataset size mismatch: Expected {expected_test}, Got {len(self.dataset_test)}")

    def get_splits(self) -> Sequence[int] | None:
        return self.split

    @abstractmethod
    def definitions(self):
        pass

    def prepare_data(self):
        pass

    @abstractmethod
    def setup(self, stage: Literal["train", "test"]):
        pass

    def update_transformations(self, transformations: Sequence[Callable]):
        """
        Update the transformations applied to the dataset.
        This is useful if you want to change the transformations after the dataset has been initialized.
        """
        datasets_not_initialized = self.dataset_train is None and self.dataset_val is None and self.dataset_test is None
        assert datasets_not_initialized, "Cannot update transformations after datasets have been initialized."
        assert isinstance(transformations, list), "Transformations must be a list of torchvision.transforms.v2.Compose objects."
        assert len(transformations) > 0, "Transformations list must not be empty."

        self.transformations = v2.Compose(transformations + self.transformations.transforms)

    def train_dataloader(self):
        if 'train' in self._loaders:
            return self._loaders['train']

        if self.dataset_train is None:
            self.setup("train")
        
        self._loaders['train'] = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=bool(self.pin_memory_device),
            pin_memory_device=self.pin_memory_device,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )
        return self._loaders['train']

    def validation_dataloader(self):
        if 'val' in self._loaders:
            return self._loaders['val']

        if self.dataset_val is None:
            self.setup("train")

        self._loaders['val'] = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=bool(self.pin_memory_device),
            pin_memory_device=self.pin_memory_device,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )
        return self._loaders['val']

    def test_dataloader(self):
        if 'test' in self._loaders:
            return self._loaders['test']

        if self.dataset_test is None:
            self.setup("test")

        self._loaders['test'] = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=bool(self.pin_memory_device),
            pin_memory_device=self.pin_memory_device,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )
        return self._loaders['test']

    def shutdown(self):
        """
        Explicitly shutdown DataLoader workers to free file descriptors.
        """
        for key, loader in self._loaders.items():
            if hasattr(loader, '_iterator') and loader._iterator is not None:
                try:
                    loader._iterator._shutdown_workers()
                except Exception:
                    pass
            del loader
        self._loaders.clear()
