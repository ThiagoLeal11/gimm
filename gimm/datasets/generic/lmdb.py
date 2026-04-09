import io
import importlib
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from gimm.datasets.definition import Dataset, _apply_transforms, _normalize_label

_lmdb = importlib.import_module('lmdb')


class LMDBDataset(Dataset):
    def __init__(
            self,
            path: str | Path,
            dims: Sequence[int],
            num_classes: int,
            batch_size: int = 1,
            num_workers: int = 0,
            dynamic_transforms: Optional[Sequence[Callable]] = None,
            map_size: int = 1 << 40,
    ):
        self.path = Path(path)
        self._backend_dims = tuple(dims)
        self._backend_num_classes = num_classes
        self.map_size = map_size
        self._env = None
        self._length = 0
        self._populated = False
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            data_dir=str(self.path),
            static_transforms=[],
            dynamic_transforms=dynamic_transforms,
            bake=False,
        )
        self._read_metadata()

    def definitions(self):
        self.dims = self._backend_dims
        self.num_classes = self._backend_num_classes
        self.split = [0, 0, 0]

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if stage == 'train':
            self.dataset_train = self
            self.dataset_val = self
        elif stage == 'test':
            self.dataset_test = self

    def _env_handle(self):
        if self._env is None:
            self.path.mkdir(parents=True, exist_ok=True)
            self._env = _lmdb.open(
                str(self.path),
                subdir=True,
                map_size=self.map_size,
                create=True,
                lock=False,
                readonly=False,
                readahead=False,
                meminit=False,
            )
        return self._env

    def _read_metadata(self):
        if not self.path.exists():
            return

        env = self._env_handle()
        with env.begin(write=False) as txn:
            populated = txn.get(b'__populated__')
            length = txn.get(b'__len__')

        self._populated = populated == b'1'
        self._length = int(length.decode()) if length else 0

    def populate(self, dataloader: DataLoader):
        if self._populated:
            return

        env = self._env_handle()
        index = 0
        with env.begin(write=True) as txn:
            for samples, labels in dataloader:
                for sample, label in zip(samples, labels):
                    buffer = io.BytesIO()
                    torch.save(
                        {
                            'sample': sample.detach().cpu(),
                            'label': _normalize_label(label),
                        },
                        buffer,
                    )
                    txn.put(f'item:{index:010d}'.encode(), buffer.getvalue())
                    index += 1

            txn.put(b'__len__', str(index).encode())
            txn.put(b'__populated__', b'1')

        self._length = index
        self._populated = True

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        env = self._env_handle()
        with env.begin(write=False) as txn:
            payload = txn.get(f'item:{idx:010d}'.encode())

        if payload is None:
            raise IndexError(idx)

        try:
            item = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
        except TypeError:
            item = torch.load(io.BytesIO(payload), map_location='cpu')
        sample = _apply_transforms(item['sample'], self.dynamic_transforms)
        return sample, item['label']


