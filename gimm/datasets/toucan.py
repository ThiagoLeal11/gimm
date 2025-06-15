# Single image dataset used for debug training.
# This image was created using the ChatGPT image generation

from torchvision import transforms
from torchvision.io import decode_image
from torch.utils.data import Dataset as TorchDataset

from gimm.datasets.definition import Dataset


class SingleImageDataset(TorchDataset):
    def __init__(self, transform=None, image_path='./gimm/datasets/toucan.jpg'):
        self.transform = transform or transforms.ToTensor()
        self.image_path = image_path

        # Loads the image
        image = decode_image(self.image_path).float() / 255.0
        final_image = self.transform(image)
        self.image = final_image

    def __len__(self):
        # Arbitrary large number to simulate a large dataset
        return 8_192

    def __getitem__(self, idx):
        return self.image, 0


class DatasetToucan(Dataset):
    def definitions(self):
        self.dims = (3, 32, 32)
        self.num_classes = 1
        self.num_workers = 1

        self.transformations = transforms.Compose(
            [
                transforms.Resize(
                    32, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop((32, 32)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def prepare_data(self):
        pass  # No preparation needed for a single image dataset

    def setup(self, stage: str):
        self.dataset_train = SingleImageDataset(transform=self.transformations)
        self.dataset_test = SingleImageDataset(transform=self.transformations)
