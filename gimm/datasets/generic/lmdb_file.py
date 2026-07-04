from pathlib import Path
from typing import Iterable, Literal

import torch
import lmdb

from gimm.datasets.definition import Dataset, Split
from gimm.datasets.generic._lmdb_dataset import LMDBTorchDataset


class LMDBDataset(Dataset):
    def definitions(self):
        dataset = self._load_lmdb_dataset()

        if dataset is None:
            if not self.is_on_baking:
                raise ValueError(
                    f"LMDBDataset requires an LMDB store under '{self.data_dir}' or explicit `dims` and `classes` for empty roots.",
                )

            self.split = [0, 0, 0]
            return

        sample, _ = dataset[0]
        self.dims = tuple(sample.shape)
        self.classes = {int(index): str(name) for index, name in dataset.classes.items()}
        stored_split = dataset.split or self._infer_split_from_path()
        self.split = self._build_split(stored_split, len(dataset))

    def setup(self, stage: Literal['train', 'test']):
        dataset = self._load_lmdb_dataset()
        if dataset is None:
            return

        stored_split = dataset.split or self._infer_split_from_path()

        if stage == 'test':
            if stored_split == 'test':
                self.dataset_test = dataset
            elif stored_split is None:
                self.dataset_test = dataset
            self._verify_split_sizes()
            return

        if stored_split == 'train':
            self.dataset_train = dataset
        elif stored_split == 'val':
            self.dataset_val = dataset
        elif stored_split is None:
            self.dataset_train = dataset
            self.dataset_val = dataset

        self._verify_split_sizes()

    def populate(self, dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]], *, split: Split):
        root = Path(self.data_dir)
        root.mkdir(parents=True, exist_ok=True)

        count = 0
        env = lmdb.open(
            str(root),
            subdir=True,
            readonly=False,
            create=True,
            lock=True,
            readahead=False,
            map_size=1 << 40,
            max_dbs=1,
        )
        try:
            with env.begin(write=True) as txn:
                for samples, labels in dataloader:
                    for sample, label in zip(samples, labels):
                        if not isinstance(sample, torch.Tensor):
                            raise TypeError(f'Expected image tensor, got {type(sample)!r}.')
                        if tuple(sample.shape) != tuple(self.dims):
                            raise ValueError(
                                f"LMDBDataset expected samples with shape {tuple(self.dims)}, got {tuple(sample.shape)}.",
                            )

                        payload = LMDBTorchDataset._serialize_sample(sample, label)
                        txn.put(self._sample_key(count), payload)
                        count += 1

                if count == 0:
                    raise RuntimeError(f"LMDBDataset populate() did not write any samples under '{self.data_dir}'.")

                metadata = {
                    'length': count,
                    'dims': tuple(self.dims),
                    'classes': dict(self.classes),
                    'split': split,
                    'sample_dtype': str(sample.dtype),
                    'label_dtype': str(label.dtype),
                    'label_shape': tuple(label.shape),
                }
                metadata_payload = bytes(LMDBTorchDataset._serialize_metadata(metadata))
                txn.put(b'__metadata__', metadata_payload)
        finally:
            env.close()

        dataset = LMDBTorchDataset(root)
        self.split = self._build_split(split, len(dataset))
        self.set_dataset(split, dataset)

    def _load_lmdb_dataset(self) -> LMDBTorchDataset | None:
        root = Path(self.data_dir)
        if not root.exists() or not root.is_dir() or not any(root.iterdir()):
            return None

        try:
            return LMDBTorchDataset(root)
        except (FileNotFoundError, RuntimeError):
            return None

    @staticmethod
    def _clone_label(label: torch.Tensor | int):
        if isinstance(label, torch.Tensor):
            return label.detach().cpu().clone()
        return torch.as_tensor(label)

    @staticmethod
    def _sample_key(index: int) -> bytes:
        return f'sample:{index:08d}'.encode('utf-8')

    @staticmethod
    def _build_split(split: Split | None, length: int) -> list[int]:
        if split == 'train':
            return [length, 0, 0]
        if split == 'val':
            return [0, length, 0]
        if split == 'test':
            return [0, 0, length]
        return [length, 0, 0]

    def _infer_split_from_path(self) -> Split | None:
        name = Path(self.data_dir).name
        if name in {'train', 'val', 'test'}:
            return name
        return None


