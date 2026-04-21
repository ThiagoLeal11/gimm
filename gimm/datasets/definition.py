from abc import ABC, abstractmethod
from functools import lru_cache
import hashlib
import json
import logging
import pathlib
import tempfile
from typing import Callable, Iterable, Literal, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset, default_collate, random_split
from tqdm.auto import tqdm
from torchvision.transforms import v2

Split = Literal['train', 'val', 'test']
Batch = tuple[torch.Tensor, torch.Tensor]

@lru_cache(maxsize=None)
def _load_bake_dataset_class(bake_type: str) -> type["Dataset"]:
    if bake_type == 'memory':
        from gimm.datasets.generic.memory import MemoryDataset
        return MemoryDataset
    if bake_type == 'lmdb':
        from gimm.datasets.generic.lmdb import LMDBDataset
        return LMDBDataset
    if bake_type == 'folder':
        from gimm.datasets.generic.folder import FolderDataset
        return FolderDataset

    raise ValueError(f"The dataset container type for baking ({bake_type}) is not supported. Valid ones are: `memory`, `lmdb` and `folder`.")


class Dataset(ABC):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 4,
        data_dir: str = ".",
        *,
        dataset_name: Optional[str] = None,
        seed: int = 42,
        shuffle: bool = True,
        prefetch_factor: int = 1,
        pin_memory_device: torch.device = 'cpu',
        persistent_workers: bool = True,
        static_transforms: Optional[list[Callable]] = None,
        dynamic_transforms: Optional[list[Callable]] = None,
        dims: Optional[Sequence[int]]  = None,
        classes: Optional[dict[int, str]] = None,
        bake: bool = False,
        bake_type: Literal['memory', 'lmdb', 'folder'] = 'memory',
        bake_path: Optional[str] = None,

        split_config: Optional[list[int | float]] = None,
    ):
        super().__init__()
        self.dataset_train: Optional[TorchDataset] = None
        self.dataset_val: Optional[TorchDataset] = None
        self.dataset_test: Optional[TorchDataset] = None
        self.split: Optional[Sequence[int]] = None
        self.split_config = split_config

        self.dataset_name = dataset_name or type(self).__name__.removeprefix('Dataset').lower()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory_device: str = str(pin_memory_device) if str(pin_memory_device).lower() != 'cpu' else ""
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

        self.dims: Sequence[int] = dims or []
        self.classes: dict[int, str] = classes or {}
        self.static_transforms = list(static_transforms) if static_transforms else []
        self.dynamic_transforms = list(dynamic_transforms) if dynamic_transforms else []
        self.bake = bake
        self.bake_type = bake_type
        self.bake_path = bake_path
        self.definitions()

        assert self.dims, "Dataset dimensions must be defined in definitions() method."
        assert self.classes, "Dataset classes must be defined in definitions() method."
        assert self.split is not None, "Dataset split must be defined in definitions() method."
        assert len(self.split) == 3, "Dataset split must be a list of three integers: [train_size, val_size, test_size]"

        if self.split_config:
            assert len(self.split_config) == 3, "split_config must be a list of three integers: [train_size, val_size, test_size]"

        if not self.static_transforms:
            self.static_transforms = [v2.ToImage()]
        if not self.dynamic_transforms:
            self.dynamic_transforms = [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.5,) * self.dims[0], (0.5,) * self.dims[0]),
            ]
            if bake and bake_type == 'memory':
                self.static_transforms += self.dynamic_transforms
                self.dynamic_transforms = []


        self.prepare_data()
        self._validation_applied = False
        self._loaders = {}
        self._baked_dataset_train: Optional[TorchDataset] = None
        self._baked_dataset_val: Optional[TorchDataset] = None
        self._baked_dataset_test: Optional[TorchDataset] = None

    def _split_train_val(self, full_train_dataset):
        split = self.get_splits()  # format: [train_size, val_size, test_size]
        train_size, val_size = split[:2]
        full_len = len(full_train_dataset)

        if train_size == -1:
            train_size = full_len - val_size
        elif val_size == -1:
            val_size = full_len - train_size

        if train_size + val_size != full_len:
             raise ValueError(f"Split sizes {train_size} + {val_size} do not match dataset length {full_len}")

        datasets = random_split(
            full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(self.seed)
        )

        self._verify_split_sizes()
        return datasets

    def _verify_split_sizes(self):
        if self.split is None:
            return

        expected_train, expected_val, expected_test = self.split
        if self.dataset_train is not None and expected_train != -1 and len(self.dataset_train) != expected_train:
            logging.warning(f"Train dataset size mismatch: Expected {expected_train}, Got {len(self.dataset_train)}")
        if self.dataset_val is not None and expected_val != -1 and len(self.dataset_val) != expected_val:
            logging.warning(f"Validation dataset size mismatch: Expected {expected_val}, Got {len(self.dataset_val)}")
        if self.dataset_test is not None and expected_test != -1 and len(self.dataset_test) != expected_test:
            logging.warning(f"Test dataset size mismatch: Expected {expected_test}, Got {len(self.dataset_test)}")

    def get_splits(self) -> Sequence[int] | None:
        return self.split

    def get_dataset(self, split: Split):
        return getattr(self, f"dataset_{split}")

    def set_baked(self, split: Split, dataset: TorchDataset):
        setattr(self, f"baked_{split}", dataset)

    @abstractmethod
    def definitions(self):
        pass

    def prepare_data(self):
        pass

    def populate(self, dataloader: Iterable[Batch], *, split: Split):
        raise NotImplementedError

    @abstractmethod
    def setup(self, stage: Literal["train", "test"]):
        pass

    def _compute_hash(self, split: Split) -> str:
        dataset = self.get_dataset(split)
        payload = {
            'dataset_name': type(self).__name__,
            'split': split,
            'dims': list(self.dims or []),
            'seed': self.seed,
            'source': {
                'type': type(dataset).__qualname__,
                'module': type(dataset).__module__,
                'size': len(dataset),
            },
            'transforms': [{
                'module': type(transform).__module__,
                'name': type(transform).__qualname__,
                'repr': repr(transform),
            } for transform in self.static_transforms],
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()[:16]

    def _resolve_bake_path(self, split: Split) -> pathlib.Path:
        transform_hash = self._compute_hash(split)
        file_name = f'{self.dataset_name}/{split}/{transform_hash}'

        template = self.bake_path
        if template is None or template == '':
            base_dir = pathlib.Path(tempfile.gettempdir()) / 'gimm' / 'baked_dataset'
            return base_dir / file_name

        placeholders = {'dataset_name', 'split', 'transform_hash'}
        if any(f'{{{placeholder}}}' in template for placeholder in placeholders):
            return pathlib.Path(template.format(
                dataset_name=self.dataset_name,
                split=split,
                transform_hash=transform_hash,
            ))

        return pathlib.Path(template) / file_name

    @staticmethod
    def _build_collate_fn(transforms: Sequence[Callable]):
        if not transforms:
            return None

        apply_transforms = v2.Compose(transforms)
        def collate_fn(batch):
            transformed_batch = [(apply_transforms(sample), label) for (sample, label) in batch]
            return default_collate(transformed_batch)

        return collate_fn

    def _generator(self) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return generator

    def build_dataloader(self, split: Split, *, shuffle: Optional[bool] = None) -> DataLoader:
        dataset = self.get_dataset(split)
        assert dataset is not None

        transforms = list(self.static_transforms + self.dynamic_transforms)
        if self.bake:
            transforms = list(self.static_transforms)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers, #if not for_baking else 0,
            shuffle=self.shuffle if shuffle is None else shuffle,
            pin_memory=bool(self.pin_memory_device),
            pin_memory_device=self.pin_memory_device,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            collate_fn=self._build_collate_fn(transforms),
            generator=self._generator(),
        )

    def _create_baked_dataset(self) -> 'Dataset':
        backend_cls = _load_bake_dataset_class(self.bake_type)
        dataset = backend_cls(
            dims=self.dims,
            classes=self.classes,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            seed=self.seed,
            shuffle=self.shuffle,
            prefetch_factor=self.prefetch_factor,
            pin_memory_device=torch.device(self.pin_memory_device or 'cpu'),
            persistent_workers=self.persistent_workers,
            dynamic_transforms=self.dynamic_transforms,
            bake=False,
            bake_type=self.bake_type,
            bake_path=self.bake_path
        )
        return dataset

    def add_static_transforms(self, transformations: list[Callable], replace: bool = False, index: int = 0):
        """
        Adds transforms before baking. This is materialized when `bake=True`
        """
        datasets_not_initialized = self.dataset_train is None and self.dataset_val is None and self.dataset_test is None
        assert datasets_not_initialized, "Cannot update transformations after datasets have been initialized."
        self.static_transforms = transformations + self.static_transforms

    def add_dynamic_transforms(self, transformations: list[Callable], replace: bool = False, index: int = -1):
        """
        Adds runtime transforms, even when `bake=True`.
        """
        datasets_not_initialized = self.dataset_train is None and self.dataset_val is None and self.dataset_test is None
        assert datasets_not_initialized, "Cannot update transformations after datasets have been initialized."
        self.dynamic_transforms = self.dynamic_transforms + transformations

    def _build_baked_loader(self, split: Literal['train', 'val', 'test'], loader: DataLoader) -> DataLoader:
        baked_dataset = self._create_baked_dataset()
        progress_loader = tqdm(
            loader,
            total=len(loader),
            desc=f'Baking {self.dataset_name} {split}',
            leave=False,
        )
        baked_dataset.populate(progress_loader, split=split)

        self.set_baked(split, baked_dataset)
        return baked_dataset.build_dataloader(split)

    def _get_or_create_loader(self, split: Literal['train', 'val', 'test']) -> DataLoader:
        # Reusing existing dataloader
        if split in self._loaders:
            return self._loaders[split]

        # Initialize dataset if not already done
        if self.get_dataset(split) is None:
            self.setup('test' if split == 'test' else 'train')

        initial_shuffle = self.shuffle if not self.bake else False
        loader = self.build_dataloader(split, shuffle=initial_shuffle)
        # Bake the dataset if needed
        if self.bake and self.static_transforms:
            print(f"Preparing baked dataset for '{split}' using {self.bake_type} backend...")
            loader = self._build_baked_loader(split, loader)

        self._loaders[split] = loader
        return loader

    def train_dataloader(self):
        return self._get_or_create_loader('train')

    def validation_dataloader(self):
        return self._get_or_create_loader('val')

    def test_dataloader(self):
        return self._get_or_create_loader('test')

    def shutdown(self):
        """
        Explicitly shutdown DataLoader workers to free file descriptors.
        """
        for key, loader in self._loaders.items():
            if hasattr(loader, '_iterator') and loader._iterator is not None:
                try:
                    loader._iterator._shutdown_workers()
                except Exception:
                    pass
            del loader
        self._loaders.clear()
