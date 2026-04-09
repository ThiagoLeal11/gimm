from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, cast

import torch
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms import v2

from gimm.datasets.definition import Dataset


class MemoryDataset(Dataset):
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
        self.samples: torch.Tensor = torch.empty((0, *self._backend_dims))
        self.labels: torch.Tensor = torch.empty((0,), dtype=torch.long)
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            data_dir=str(self.path),
            static_transforms=[],
            dynamic_transforms=dynamic_transforms,
            bake=False,
        )

    def definitions(self):
        self.dims = self._backend_dims
        self.num_classes = self._backend_num_classes
        self.split = [0, 0, 0]

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if stage == 'train':
            baked_self = cast(TorchDataset, cast(object, self))
            self.dataset_train = baked_self
            self.dataset_val = baked_self
        elif stage == 'test':
            self.dataset_test = cast(TorchDataset, cast(object, self))

    def populate(self, dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]]):
        samples_buffer = []
        labels_buffer = []
        for samples, labels in dataloader:
            samples_buffer.append(samples.detach().cpu().clone())
            labels_buffer.append(labels.detach().cpu().clone())

        if samples_buffer:
            self.samples = torch.cat(samples_buffer, dim=0)
            self.labels = torch.cat(labels_buffer, dim=0)

        baked_self = cast(TorchDataset, cast(object, self))
        self.dataset_train = baked_self
        self.dataset_val = baked_self
        self.dataset_test = baked_self

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample, self.labels[idx]
