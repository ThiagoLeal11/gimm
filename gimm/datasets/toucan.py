# Single image dataset used for debug training.
# This image was created using the ChatGPT image generation

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset as TorchDataset

from gimm.datasets.definition import Dataset


class SingleImageDataset(TorchDataset):
    def __init__(self, transform=None, image_path='./gimm/datasets/losing_toucan.jpg'):
        self.transform = transform or transforms.ToTensor()
        self.image_path = image_path

        # Carrega a imagem
        image = Image.open(self.image_path).convert('RGB')
        final_image = self.transform(image)
        # self.image = final_image.unsqueeze(0)
        self.image = final_image

    def __len__(self):
        return 42_000  # Arbitrary large number to simulate a large dataset

    def __getitem__(self, idx):
        return self.image, 0


class DatasetToucan(Dataset):
    def definitions(self):
        self.dims = (3, 32, 32)
        self.num_classes = 1
        self.num_workers = 1

        self.dataset_transform = transforms.Compose([
            transforms.Resize(32, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def prepare_data(self):
        pass  # No preparation needed for a single image dataset

    def setup(self, stage: str):
        self.dataset_train = SingleImageDataset(transform=self.dataset_transform)
        self.dataset_test = SingleImageDataset(transform=self.dataset_transform)