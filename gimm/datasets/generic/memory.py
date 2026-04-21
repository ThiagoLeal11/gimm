from typing import Iterable, cast

import torch
from torch.utils.data import Dataset as TorchDataset
from gimm.datasets.definition import Dataset, Batch, Split


class MemoryDataset(Dataset):
    def definitions(self):
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

    def populate(self, dataloader: Iterable[Batch], *, split: Split):
        samples_buffer = []
        labels_buffer = []
        for samples, labels in dataloader:
            samples_buffer.append(samples.detach().cpu().clone())
            labels_buffer.append(labels.detach().cpu().clone())

        if samples_buffer:
            self._buffer['samples'] = torch.cat(samples_buffer, dim=0)
            self._buffer['labels'] = torch.cat(labels_buffer, dim=0)

        baked_self = cast(TorchDataset, cast(object, self))
        self.dataset_train = baked_self
        self.dataset_val = baked_self
        self.dataset_test = baked_self

    def __len__(self):
        return len(self._buffer['samples'])

    def __getitem__(self, idx):
        sample = self._buffer['samples'][idx]
        return sample, self._buffer['labels'][idx]
