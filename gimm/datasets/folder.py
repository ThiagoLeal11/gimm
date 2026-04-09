import json
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from gimm.datasets.definition import Dataset


class FolderDataset(Dataset):
    """Mock folder-backed baked dataset.

    For now this stores each sample as a `.pt` file plus a small metadata.json.
    """

    def __init__(
            self,
            path: str | Path,
            dims: Sequence[int],
            num_classes: int,
            batch_size: int = 1,
            num_workers: int = 0,
            dynamic_transforms: Optional[Sequence[Callable]] = None,
    ):
        self.path = Path(path)
        self._backend_dims = tuple(dims)
        self._backend_num_classes = num_classes
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

    def _metadata_path(self) -> Path:
        return self.path / 'metadata.json'

    def _sample_path(self, idx: int) -> Path:
        return self.path / f'{idx:010d}.pt'

    def _read_metadata(self):
        metadata_path = self._metadata_path()
        if not metadata_path.exists():
            return

        metadata = json.loads(metadata_path.read_text())
        self._length = metadata.get('length', 0)
        self._populated = metadata.get('populated', False)

    def populate(self, dataloader: DataLoader):
        if self._populated:
            return

        self.path.mkdir(parents=True, exist_ok=True)
        index = 0
        for samples, labels in dataloader:
            for sample, label in zip(samples, labels):
                torch.save(
                    {
                        'sample': sample.detach().cpu(),
                        'label': label.detach().cpu(),
                    },
                    self._sample_path(index),
                )
                index += 1

        self._length = index
        self._populated = True
        self._metadata_path().write_text(json.dumps({'length': index, 'populated': True}))

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        sample_path = self._sample_path(idx)
        if not sample_path.exists():
            raise IndexError(idx)

        try:
            item = torch.load(sample_path, map_location='cpu', weights_only=False)
        except TypeError:
            item = torch.load(sample_path, map_location='cpu')
        sample = item['sample']
        return sample, item['label']


