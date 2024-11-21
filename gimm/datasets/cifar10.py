from typing import Literal

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

from gimm.datasets.definition import Dataset


class DatasetCifar10(Dataset):
    def definitions(self):
        self.dims = (3, 32, 32)
        self.num_classes = 10

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Literal["train", "test"]):
        if stage == "train":
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.dataset_train, self.dataset_val = random_split(cifar_full, [45000, 5000])

        elif stage == "test":
            self.dataset_test = CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )


class DatasetCifar10PL(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 1,
        data_dir: str = ".",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dims = (3, 32, 32)
        self.num_classes = 10

        self.transform = transforms.ToTensor()

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
