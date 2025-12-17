from typing import Literal

from torchvision.datasets import CIFAR10

from gimm.datasets.definition import Dataset


class DatasetCifar10(Dataset):
    def definitions(self):
        self.dims = (3, 32, 32)
        self.num_classes = 10
        self.split = [45000, 5000, 10000]

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Literal["train", "test"]):
        if stage == "train":
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transformations)
            self.dataset_train, self.dataset_val = self._split_train_val(cifar_full)

        elif stage == "test":
            self.dataset_test = CIFAR10(
                self.data_dir, train=False, transform=self.transformations
            )
