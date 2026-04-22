# Single image dataset used for debug training.
# This image was created using the ChatGPT image generation

import pathlib
from typing import Literal

from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset as TorchDataset

from gimm.datasets.definition import Dataset


class SingleImageDataset(TorchDataset):
    def __init__(self, transform=None, image_path='./gimm/datasets/toucan.jpg'):
        self.transform = transform or transforms.ToTensor()
        self.image_path = image_path

        # Loads the image
        image = read_image(self.image_path).float() / 255.0
        final_image = self.transform(image)
        self.image = final_image

    def __len__(self):
        # Arbitrary large number to simulate a large dataset
        return 8_192

    def __getitem__(self, idx):
        return self.image, 0


class DatasetToucan(Dataset):
    def definitions(self):
        self.dims = (3, 1024, 1024)
        self.classes = {0: 'toucan'}
        self.split = [8192, 8192, 8192]
        self.num_workers = 1

    def prepare_data(self):
        pass  # No preparation needed for a single image dataset

    def setup(self, stage: Literal["train", "test"]):
        image_path = pathlib.Path(__file__).with_name('toucan.jpg')
        trans = transforms.Compose(self.static_transforms)

        if stage == 'train':
            self.dataset_train = SingleImageDataset(image_path=str(image_path), transform=trans)
            self.dataset_val = SingleImageDataset(image_path=str(image_path), transform=trans)

        elif stage == 'test':
            self.dataset_test = SingleImageDataset(image_path=str(image_path), transform=trans)
