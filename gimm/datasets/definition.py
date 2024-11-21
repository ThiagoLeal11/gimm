from abc import ABC, abstractmethod
from typing import Literal

from torch.utils.data import DataLoader
from torchvision import transforms


class Dataset(ABC):
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
        )

    def val_dataloader(self):
        if self.dataset_val is None:
            self.setup("train")

        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        if self.dataset_test is None:
            self.setup("test")

        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
