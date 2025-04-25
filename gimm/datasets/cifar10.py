from typing import Literal

from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

from gimm.datasets.definition import Dataset


class DatasetCifar10(Dataset):
    def definitions(self):
        self.dims = (3, 32, 32)
        self.num_classes = 10

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

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
