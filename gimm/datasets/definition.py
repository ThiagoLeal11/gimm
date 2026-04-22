from abc import ABC, abstractmethod
from functools import lru_cache
import hashlib
import json
import logging
import pathlib
import tempfile
from typing import Callable, Iterable, Literal, Optional, Sequence, Sized, cast, Any

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
        bake_min_size: int = 0,

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
        self.bake_min_size = bake_min_size
        assert self.bake_min_size >= 0, "bake_min_size must be >= 0. Use 0 to disable duplication."
        self._buffer: dict[str, Any] = {}
        self.definitions()

        if self.bake_path not in (None, ''):
            assert pathlib.Path(self.data_dir) != pathlib.Path(self.bake_path), "Baked path root must be different from data_dir root to avoid accidental data loss."

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

    @staticmethod
    def _compute_split(full_dataset, split: Sequence[int | float]) -> list[int]:
        full_len = len(full_dataset)
        split_sizes = list(split)
        unknown_count = split_sizes.count(-1)

        if any(isinstance(size, float) for size in split_sizes):
            if unknown_count:
                raise ValueError("Percentage split cannot contain -1 values.")

            total = float(sum(split_sizes))
            if abs(total - 1.0) > 1e-8:
                raise ValueError(f"Percentage split must sum to 1.0, got {total}")

            if any(size < 0 for size in split_sizes):
                raise ValueError("Split values must be non-negative.")

            split_sizes = [int(full_len * float(size)) for size in split_sizes[:-1]]
            split_sizes.append(full_len - sum(split_sizes))

        if unknown_count > 1:
            raise ValueError("Split can contain at most one -1 value.")

        if unknown_count == 1:
            known_total = sum(size for size in split_sizes if size != -1)
            missing_size = full_len - known_total
            if missing_size < 0:
                raise ValueError(f"Split sizes {split_sizes} exceed dataset length {full_len}")
            split_sizes[split_sizes.index(-1)] = missing_size

        if sum(split_sizes) != full_len:
            raise ValueError(f"Split sizes {split_sizes} do not match dataset length {full_len}")

        return split_sizes

    def _split_dataset(self, full_dataset, split: Sequence[int | float]):
        split_sizes = self._compute_split(full_dataset, split)
        datasets = random_split(
            full_dataset, split_sizes, generator=self._generator()
        )

        self._verify_split_sizes()
        return datasets

    def _verify_split_sizes(self):
        if self.split is None:
            return

        expected_train, expected_val, expected_test = self.split
        train_dataset = cast(Optional[Sized], self.dataset_train)
        val_dataset = cast(Optional[Sized], self.dataset_val)
        test_dataset = cast(Optional[Sized], self.dataset_test)

        if train_dataset is not None and expected_train != -1 and len(train_dataset) != expected_train:
            logging.warning(f"Train dataset size mismatch: Expected {expected_train}, Got {len(train_dataset)}")
        if val_dataset is not None and expected_val != -1 and len(val_dataset) != expected_val:
            logging.warning(f"Validation dataset size mismatch: Expected {expected_val}, Got {len(val_dataset)}")
        if test_dataset is not None and expected_test != -1 and len(test_dataset) != expected_test:
            logging.warning(f"Test dataset size mismatch: Expected {expected_test}, Got {len(test_dataset)}")

    def get_splits(self) -> Sequence[int] | None:
        return self.split

    def get_num_classes(self) -> int:
        return len(self.classes)

    def get_dataset(self, split: Split):
        return getattr(self, f"dataset_{split}")

    def set_dataset(self, split: Split, dataset: TorchDataset):
        setattr(self, f"dataset_{split}", dataset)

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
            'bake_min_size': self.bake_min_size,
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

    def _create_baked_dataset(self, split: Split) -> 'Dataset':
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
            data_dir=str(pathlib.Path(self.bake_path) / split),
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

    def _build_baked_loader(self, split: Split, loader: DataLoader) -> DataLoader:
        baked_dataset = self._create_baked_dataset(split)
        repetitions = self._get_bake_repetitions(split)
        progress_loader = tqdm(
            self._repeat_loader(loader, repetitions),
            total=len(loader) * repetitions,
            desc=f"Baking {self.dataset_name} {split}" + (f" x{repetitions}" if repetitions > 1 else ''),
            leave=False,
        )
        baked_dataset.populate(progress_loader, split=split)

        self.set_baked(split, baked_dataset)
        return baked_dataset.build_dataloader(split)

    def _get_bake_repetitions(self, split: Split) -> int:
        if split != 'train' or self.bake_min_size <= 0:
            return 1

        dataset = self.get_dataset(split)
        dataset_size = len(dataset)
        if dataset_size <= 0:
            return 1

        return max(1, (self.bake_min_size + dataset_size - 1) // dataset_size)

    @staticmethod
    def _repeat_loader(loader: DataLoader, repetitions: int):
        for _ in range(repetitions):
            yield from loader

    def _get_or_create_loader(self, split: Split) -> DataLoader:
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
            print(f"Baked dataset for '{split}' is done!")

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
