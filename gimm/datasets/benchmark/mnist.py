from typing import Literal

from torchvision.datasets import MNIST

from gimm.datasets.definition import Dataset


class DatasetMNIST(Dataset):
    def definitions(self):
        self.dims = (1, 28, 28)
        self.classes = {index: name for index, name in enumerate(MNIST.classes)}
        self.split = [55000, 5000, 10000]

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Literal["train", "test"]):
        if stage == "train":
            mnist_full = MNIST(self.data_dir, train=True)
            train_size, val_size, _ = self.get_splits()
            self.dataset_train, self.dataset_val = self._split_dataset(mnist_full, [train_size, val_size])

        # Assign test dataset for use in dataloader(s)
        elif stage == "test":
            self.dataset_test = MNIST(self.data_dir, train=False)
