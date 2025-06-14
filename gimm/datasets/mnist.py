from typing import Literal

from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from gimm.datasets.definition import Dataset


class DatasetMNIST(Dataset):
    def definitions(self):
        self.dims = (1, 28, 28)
        self.num_classes = 10

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Literal["train", "test"]):
        if stage == "train":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.dataset_train, self.dataset_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.dataset_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )
