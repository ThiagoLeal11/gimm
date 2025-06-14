from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch.utils.data import DataLoader
from torchvision import transforms


class Dataset(ABC):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 1,
        data_dir: str = ".",
        shuffle: bool = True,
        pin_memory_device: torch.device = 'cpu',
            # TODO: permitir que isso seja uma configuração
            # TODO: remover pin_memory
        # transform: transforms.Compose = transforms.Compose([transforms.ToTensor()])
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory_device: str = str(pin_memory_device) if str(pin_memory_device).lower() != 'cpu' else ""

        self.dims = None
        self.num_classes = None
        self.transform = transforms.ToTensor()
        self.definitions()

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
            # TODO: Validar se essa é mesmo a melhor configuração.
            prefetch_factor=1,
            persistent_workers=True,
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
        )
