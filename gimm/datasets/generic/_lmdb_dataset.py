import io
from pathlib import Path
from typing import Sequence

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


class LMDBTorchDataset(TorchDataset):
    _METADATA_KEY = b'__metadata__'

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._env = None
        metadata = self._read_metadata()
        self.dims = tuple(metadata['dims'])
        self.classes = {int(index): name for index, name in metadata['classes'].items()}
        self.length = int(metadata['length'])
        self.split = metadata.get('split')
        self.sample_dtype = self._dtype_from_name(metadata['sample_dtype'])
        self.label_dtype = self._dtype_from_name(metadata['label_dtype'])
        self.label_shape = tuple(metadata['label_shape'])

        self._label_size = self._num_bytes(self.label_shape, self.label_dtype)
        self._label_np_dtype = self._numpy_dtype_from_torch(self.label_dtype)
        self._sample_np_dtype = self._numpy_dtype_from_torch(self.sample_dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self._get_env().begin(write=False) as txn:
            payload = txn.get(self._sample_key(idx))

        if payload is None:
            raise IndexError(f'LMDB entry {idx} was not found under {self.root!r}.')

        return self._deserialize_sample(payload)

    def __getitems__(self, indices: Sequence[int]):
        if not indices:
            return []

        keys = [self._sample_key(idx) for idx in indices]
        order_map = {key: pos for pos, key in enumerate(keys)}
        restored_order: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(keys)

        with self._get_env().begin(write=False) as txn:
            with txn.cursor() as cur:
                results = cur.getmulti(keys)
                for key, payload in results:
                    safe_key = self._to_bytes(key)
                    restored_order[order_map[safe_key]] = self._deserialize_sample(self._to_bytes(payload))

            if len(results) != len(keys):
                raise IndexError(f'LMDB entries {list(indices)} were not found under {self.root!r}.')

        return restored_order

    def _read_metadata(self):
        env = self._open_env(readonly=True)
        try:
            with env.begin(write=False) as txn:
                payload = txn.get(self._METADATA_KEY)
        finally:
            env.close()

        if payload is None:
            raise RuntimeError(f"LMDB metadata is missing under '{self.root}'.")

        return self._deserialize_metadata(payload)

    def _get_env(self):
        if self._env is None:
            self._env = self._open_env(readonly=True)
        return self._env

    def _open_env(self, *, readonly: bool):
        if not self.root.exists() or not self.root.is_dir():
            raise FileNotFoundError(f"LMDB root '{self.root}' does not exist.")

        return lmdb.open(
            str(self.root),
            subdir=True,
            readonly=readonly,
            create=not readonly,
            lock=not readonly,
            readahead=False,
            map_size=1 << 40,
            max_dbs=1,
        )

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_env'] = None
        return state

    def __del__(self):
        self.close()

    @staticmethod
    def _serialize_sample(sample: torch.Tensor, label: torch.Tensor) -> bytes:
        sample_bytes = sample.numpy().tobytes()
        label_bytes = label.numpy().tobytes()
        return label_bytes + sample_bytes

    @staticmethod
    def _serialize_metadata(value) -> bytes:
        buffer = io.BytesIO()
        torch.save(value, buffer)
        return buffer.getvalue()

    @staticmethod
    def _deserialize_metadata(payload: bytes | memoryview):
        buffer = io.BytesIO(LMDBTorchDataset._to_bytes(payload))
        return torch.load(buffer, map_location='cpu', weights_only=False)

    def _deserialize_sample(self, payload: bytes | memoryview):
        raw = np.frombuffer(payload, dtype=np.uint8).copy()

        label = torch.from_numpy(raw[:self._label_size].view(self._label_np_dtype).reshape(self.label_shape))
        sample = torch.from_numpy(raw[self._label_size:].view(self._sample_np_dtype).reshape(self.dims))
        return sample.clone(), label.clone()

    @staticmethod
    def _dtype_from_name(name: str) -> torch.dtype:
        return getattr(torch, name.removeprefix('torch.'))

    @staticmethod
    def _numpy_dtype_from_torch(dtype: torch.dtype):
        return {
            torch.uint8: np.uint8,
            torch.int8: np.int8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.bool: np.bool_,
        }[dtype]

    @staticmethod
    def _num_bytes(shape: Sequence[int], dtype: torch.dtype) -> int:
        numel = 1
        for dim in shape:
            numel *= dim
        return numel * torch.empty((), dtype=dtype).element_size()

    @staticmethod
    def _to_bytes(payload: bytes | memoryview) -> bytes:
        if isinstance(payload, memoryview):
            return payload.tobytes()
        return payload

    @staticmethod
    def _sample_key(index: int) -> bytes:
        return f'sample:{index:08d}'.encode('utf-8')

