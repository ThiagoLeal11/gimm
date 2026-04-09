import glob
import os
import re
import zipfile
from typing import Literal, List

import requests
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms
from torchvision.io import decode_image

from gimm.datasets.definition import Dataset


class DatasetPistachio1(Dataset):
    def definitions(self):
        self.dims = (3, 32, 32)
        self.num_classes = 1

        self.static_transforms = [
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BILINEAR),
        ]

    def prepare_data(self):
        _PISTACHIO_1(self.data_dir, download=True)

    def setup(self, stage: Literal["train", "test"]):
        dataset_root_path = os.path.join(self.data_dir, 'pistachio-image-dataset')
        image_folder_path = os.path.join(dataset_root_path, 'Pistachio_Image_Dataset', 'Kirmizi_Pistachio')

        if not os.path.exists(image_folder_path):
            raise FileNotFoundError(f"Dataset not found at {image_folder_path}. Please ensure it is downloaded and extracted correctly.")

        all_files = glob.glob(os.path.join(image_folder_path, '*'))
        train_pattern = re.compile(r"kirmizi \d+\.jpg", re.IGNORECASE)
        test_pattern = re.compile(r"kirmizi \(\d+\)\.jpg", re.IGNORECASE)

        train_image_paths = []
        test_image_paths = []
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            if train_pattern.fullmatch(file_name):
                train_image_paths.append(file_path)
            elif test_pattern.fullmatch(file_name):
                test_image_paths.append(file_path)

        if not train_image_paths and not test_image_paths:
            raise FileNotFoundError(f"No valid images found in {image_folder_path}. Please check the dataset structure.")

        self.dataset_train = PistachioImageDataset(image_paths=train_image_paths)
        self.dataset_train = PistachioImageDataset(image_paths=train_image_paths)
        self.dataset_test = PistachioImageDataset(image_paths=test_image_paths)


def _PISTACHIO_1(path: str, download: bool = True):
    dataset_folder_name = 'pistachio-image-dataset'
    zip_file_name = 'pistachio-image-dataset.zip'
    zip_file_path = os.path.join(path, zip_file_name)
    extracted_dataset_path = os.path.join(path, dataset_folder_name)
    validation_path = os.path.join(extracted_dataset_path, 'Pistachio_Image_Dataset', 'Kirmizi_Pistachio')
    url = 'https://www.muratkoklu.com/datasets/vtdhnd12.php'

    # 3. Validação se já foi extraído e contém arquivos
    already_extracted = False
    if os.path.isdir(validation_path):
        if len(glob.glob(os.path.join(validation_path, '*'))) > 1000:
            already_extracted = True
            print(f"Files already downloaded and verified")

    if not already_extracted:
        # 1. Validar se o .zip já foi baixado
        zip_exists = os.path.exists(zip_file_path)

        if not zip_exists:
            if download:
                print(f"Downloading dataset")
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status() # Verifica se houve erro no request
                    with open(zip_file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"Finished downloading {zip_file_name} to '{path}'")
                    zip_exists = True
                except requests.exceptions.RequestException as e:
                    raise FileNotFoundError(f"Erros when trying to download dataset. Error: {e}")
            else:
                raise FileNotFoundError(
                    f"Dataset {zip_file_name} not founded on '{path}' and download is disabled. Please enable download or provide the dataset manually."
                )

        # 2. Tentar unzipar o arquivo
        if zip_exists: # Se o zip existe (ou foi baixado agora)
            print(f"Unzipping dataset")
            try:
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    # Extrai para uma pasta temporária primeiro para poder renomear
                    temp_extract_path = os.path.join(path, "temp_pistachio_extract")
                    zip_ref.extractall(temp_extract_path)

                    # A estrutura dentro do zip pode ser 'Pistachio_Image_Dataset/...'
                    # Precisamos garantir que o nome final da pasta seja 'pistachio-image-dataset'
                    extracted_folders = os.listdir(temp_extract_path)
                    if len(extracted_folders) == 1 and os.path.isdir(os.path.join(temp_extract_path, extracted_folders[0])):
                        # Renomeia a pasta extraída para o nome desejado
                        os.rename(os.path.join(temp_extract_path, extracted_folders[0]), extracted_dataset_path)
                        os.rmdir(temp_extract_path) # Remove a pasta temporária vazia
                    else: # Caso inesperado, apenas move o conteúdo
                        os.rename(temp_extract_path, extracted_dataset_path)

                print(f"Finished extracting")

                # 3. Revalidação após extração
                if os.path.isdir(validation_path) and len(glob.glob(os.path.join(validation_path, '*'))) > 1000:
                    print("Dataset extracted successfully and validated.")
                else:
                    raise FileNotFoundError(
                        f"Dataset {zip_file_name} was extracted but validation failed. Please check the contents of the extracted folder."
                    )
            except zipfile.BadZipFile:
                raise zipfile.BadZipFile(f"File '{zip_file_path}' is not a valid zip file or is corrupted.")
            except Exception as e:
                raise Exception(f"Error extracting dataset: {e}")
    return extracted_dataset_path


# Coloque esta classe antes da definição de DatasetPistachio1
class PistachioImageDataset(TorchDataset):
    def __init__(self, image_paths: List[str]):
        self.image_paths = image_paths
        # Como self.num_classes = 1 para DatasetPistachio1, o label será sempre 0.

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = decode_image(img_path).float() / 255.0
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo de imagem não encontrado: {img_path}")
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar a imagem {img_path}: {e}")

        # Label único (0) para a classe Kirmizi_Pistachio, pois num_classes = 1
        label = 0
        return image, label
