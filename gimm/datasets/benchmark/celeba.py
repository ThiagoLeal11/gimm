from typing import Literal

from torchvision.datasets import CelebA

from gimm.datasets.definition import Dataset


class DatasetCelebA(Dataset):
    def definitions(self):
        self.dims = (3, 178, 218)
        self.classes = {0: 'face'}
        # CelebA official split: 162770 train, 19867 val, 19962 test
        self.split = [162770, 19867, 19962]

    def prepare_data(self):
        CelebA(self.data_dir, split="train", download=True)
        CelebA(self.data_dir, split="valid", download=True)
        CelebA(self.data_dir, split="test", download=True)

    def setup(self, stage: Literal["train", "test"]):
        if stage == "train":
            self.dataset_train = CelebA(self.data_dir, split="train")
            self.dataset_val = CelebA(self.data_dir, split="valid")

        elif stage == "test":
            self.dataset_test = CelebA(self.data_dir, split="test")

