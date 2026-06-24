from pathlib import Path
from typing import Literal

from torch.utils.data import ConcatDataset
from torchvision.datasets import LSUN

from gimm.datasets.definition import Dataset


class DatasetLSUNBedrooms(Dataset):
    train_class = 'bedroom_train'
    val_class = 'bedroom_val'
    default_resolution = 256

    def definitions(self):
        self.dims = (3, self.default_resolution, self.default_resolution)
        self.classes = {0: 'bedroom'}
        self.split = [-1, -1, -1]

    def prepare_data(self):
        self._ensure_data_files_present()

        dataset_train = self._load_split(self.train_class)
        dataset_val = self._load_split(self.val_class)
        self._buffer['dataset_train_full'] = dataset_train
        self._buffer['dataset_val_full'] = dataset_val

        if self.split_config:
            dataset_full = ConcatDataset([dataset_train, dataset_val])
            self._buffer['dataset_full'] = dataset_full
            self.split = self._compute_split(dataset_full, self.split_config)
            return

        val_size = len(dataset_val) // 2
        test_size = len(dataset_val) - val_size
        self.split = [len(dataset_train), val_size, test_size]

    def setup(self, stage: Literal['train', 'test']):
        if self.split_config:
            self._setup_custom_split()
            return

        self._setup_official_split()

    def _setup_custom_split(self):
        if self.dataset_train is not None and self.dataset_val is not None and self.dataset_test is not None:
            return

        dataset_full = self._buffer.get('dataset_full')
        if dataset_full is None:
            dataset_full = ConcatDataset([
                self._load_split(self.train_class),
                self._load_split(self.val_class),
            ])
            self._buffer['dataset_full'] = dataset_full

        self.dataset_train, self.dataset_val, self.dataset_test = self._split_dataset(dataset_full, self.split)
        self._verify_split_sizes()

    def _setup_official_split(self):
        if self.dataset_train is None:
            self.dataset_train = self._buffer.get('dataset_train_full') or self._load_split(self.train_class)

        if self.dataset_val is None or self.dataset_test is None:
            dataset_val = self._buffer.get('dataset_val_full') or self._load_split(self.val_class)
            _, val_size, test_size = self.split
            self.dataset_val, self.dataset_test = self._split_dataset(dataset_val, [val_size, test_size])

        self._verify_split_sizes()

    def _load_split(self, split_name: str) -> LSUN:
        return LSUN(self.data_dir, classes=[split_name])

    def _ensure_data_files_present(self):
        root = Path(self.data_dir)
        expected_paths = {
            self.train_class: root / f'{self.train_class}_lmdb',
            self.val_class: root / f'{self.val_class}_lmdb',
        }
        missing = [path for path in expected_paths.values() if not path.exists()]
        if not missing:
            return

        missing_items = '\n'.join(f'  - {path}' for path in missing)
        raise FileNotFoundError(
            f"LSUN Bedrooms dataset not found under '{self.data_dir}'. Missing paths:\n"
            f"{missing_items}\n"
            "torchvision.datasets.LSUN does not support automatic download.\n"
            "Download LSUN Bedrooms train/val LMDB archives manually from official LSUN release or mirror, then extract them to:\n"
            f"  - {expected_paths[self.train_class]}\n"
            f"  - {expected_paths[self.val_class]}\n"
            "Expected class names are `bedroom_train` and `bedroom_val`."
        )
