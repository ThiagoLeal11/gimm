import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Literal

import torch
from torchvision.datasets import ImageFolder
from torchvision.io import read_image, write_png

from gimm.datasets.definition import Dataset, Split, Batch


class FolderDataset(Dataset):
    def definitions(self):
        is_on_baking = bool(self.dims) and bool(self.classes) and self.split_config is None
        dataset = self._load_image_folder()

        if dataset is None:
            if not is_on_baking:
                raise ValueError(
                    f"FolderDataset requires images under '{self.data_dir}' or explicit `dims` and `classes` for empty roots.",
                )

            self.split = [0, 0, 0]
            return

        sample, _ = dataset[0]
        self.dims = tuple(sample.shape)
        self.classes = {idx: name for name, idx in dataset.class_to_idx.items()}

        if not self.split_config:
            logging.warning(
                f"FolderDataset split_config missing for '{self.data_dir}'. Using default 80% / 10% / 10% split.",
            )
            self.split_config = [0.8, 0.1, 0.1]

        self.split = self._compute_split(dataset, self.split_config)
        self._buffer['dataset_full'] = dataset

    def setup(self, stage: Literal["train", "test"]):
        # Setup was already completed in a previous step
        if self.dataset_train is not None and stage == 'train' or self.dataset_test is not None and stage == 'test':
            return

        train, val, test = self._split_dataset(self._buffer.pop('dataset_full'), self.split)
        self.dataset_train = train
        self.dataset_val = val
        self.dataset_test = test

        self._verify_split_sizes()

    def populate(self, dataloader: Iterable[Batch], *, split: Split):
        split_root = Path(self.data_dir)
        split_root.mkdir(parents=True, exist_ok=True)

        counters: dict[str, int] = defaultdict(int)
        for samples, labels in dataloader:
            for sample, label in zip(samples, labels):
                # Ensure class folder
                class_name = self.classes[self._label_to_int(label)]
                class_dir = split_root / class_name
                class_dir.mkdir(parents=True, exist_ok=True)

                # Save image
                counters[class_name] += 1
                output_path = class_dir / f'{counters[class_name]:05d}.png'
                write_png(self.to_int8_image(sample), str(output_path))

        dataset = self._load_image_folder()
        if dataset is None:
            raise RuntimeError(f"FolderDataset populate() did not create any readable images under '{self.data_dir}'.")

        self.set_dataset(split, dataset)

    def _load_image_folder(self) -> ImageFolder | None:
        root = Path(self.data_dir)
        if not root.exists() or not root.is_dir():
            return None

        class_dirs = [
            entry
            for entry in root.iterdir()
            if entry.is_dir() and not entry.name.startswith('.')
        ]
        if not class_dirs:
            return None

        try:
            return ImageFolder(self.data_dir, loader=read_image)
        except FileNotFoundError:
            return None

    @staticmethod
    def _label_to_int(label: torch.Tensor | int) -> int:
        if isinstance(label, torch.Tensor):
            return int(label.detach().cpu().item())
        return int(label)

    def to_int8_image(self, sample: torch.Tensor) -> torch.Tensor:
        if not isinstance(sample, torch.Tensor):
            raise TypeError(f'Expected image tensor, got {type(sample)!r}.')

        image = sample.detach().cpu()
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if image.ndim != 3:
            raise ValueError(f'Expected image tensor with shape [C, H, W], got {tuple(image.shape)}.')

        if image.dtype == torch.uint8:
            return image.contiguous()

        image = image.float()
        if not torch.isfinite(image).all():
            raise ValueError('Image tensor contains non-finite values and cannot be written as PNG.')

        min_value = float(image.min().item())
        max_value = float(image.max().item())

        if -1.0 <= min_value and max_value <= 1.0:
            if min_value < 0.0:
                image = ((image + 1.0) * 127.5).round()
            else:
                image = (image * 255.0).round()
        elif 0.0 <= min_value and max_value <= 255.0:
            image = image.round()
        else:
            raise ValueError(
                f'Unsupported image range [{min_value}, {max_value}]. Expected uint8-like [0, 255], float [0, 1], or float [-1, 1].',
            )

        return image.clamp(0.0, 255.0).to(torch.uint8).contiguous()
