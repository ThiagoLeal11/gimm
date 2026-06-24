import io
import importlib
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset

_lmdb = importlib.import_module('lmdb')


class LMDBTorchDataset(TorchDataset):
    _METADATA_KEY = b'__metadata__'

    def __init__(self, root: str | Path):
        self.root = Path(root)
        metadata = self._read_metadata()
        self.dims = tuple(metadata['dims'])
        self.classes = {int(index): name for index, name in metadata['classes'].items()}
        self.length = int(metadata['length'])
        self.split = metadata.get('split')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        length = len(self)
        if idx < 0:
            idx += length
        if idx < 0 or idx >= length:
            raise IndexError(f'Index {idx} is out of range for LMDBTorchDataset of size {length}.')

        env = self._open_env(readonly=True)
        try:
            with env.begin(write=False) as txn:
                payload = txn.get(self._sample_key(idx))
        finally:
            env.close()

        if payload is None:
            raise IndexError(f'LMDB entry {idx} was not found under {self.root!r}.')

        return self._deserialize(payload)

    def _read_metadata(self):
        env = self._open_env(readonly=True)
        try:
            with env.begin(write=False) as txn:
                payload = txn.get(self._METADATA_KEY)
        finally:
            env.close()

        if payload is None:
            raise RuntimeError(f"LMDB metadata is missing under '{self.root}'.")

        return self._deserialize(payload)

    def _open_env(self, *, readonly: bool):
        if not self.root.exists() or not self.root.is_dir():
            raise FileNotFoundError(f"LMDB root '{self.root}' does not exist.")

        return _lmdb.open(
            str(self.root),
            subdir=True,
            readonly=readonly,
            create=not readonly,
            lock=not readonly,
            readahead=False,
            map_size=1 << 40,
            max_dbs=1,
        )

    @staticmethod
    def _serialize(value) -> bytes:
        buffer = io.BytesIO()
        torch.save(value, buffer)
        return buffer.getvalue()

    @staticmethod
    def _deserialize(payload: bytes):
        buffer = io.BytesIO(payload)
        return torch.load(buffer, map_location='cpu', weights_only=False)

    @staticmethod
    def _sample_key(index: int) -> bytes:
        return f'sample:{index:08d}'.encode('utf-8')

