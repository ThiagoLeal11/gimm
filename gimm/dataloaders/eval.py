from typing import Iterator, Literal, Optional, Generator, Iterable
from itertools import chain

import torch
from torch.utils.data import DataLoader

from gimm.datasets.definition import Dataset

DL = torch.utils.data.DataLoader
Batch = tuple[torch.Tensor, torch.Tensor]
Policy = Literal['repeat', 'supplement', 'strict']


def validate_policy(dataset: Dataset, target_samples: int = 20_000, policy: Policy = 'strict'):
    train, val, test = dataset.get_splits()
    if policy == 'strict':
        assert target_samples <= val, f"Strict policy requires validation set to have at least target_samples, got {val}"
    if policy == 'supplement':
        assert target_samples <= (val + train), f"Supplement policy requires combined train and val sets to have at least target_samples, got {val + train}"


class ValidationLoader(Iterable):
    def __init__(
            self,
            dataloader: DataLoader,
            supplement_dataloader: Optional[DataLoader] = None,
            target_samples: int = 20_000,
            policy: Policy = 'strict',
    ):
        self.dataloader = dataloader
        self.supplement_dataloader = supplement_dataloader
        self.batch_size = dataloader.batch_size
        self.target_samples = target_samples
        self.policy = policy

        assert target_samples > 0, "target_samples must be positive"

    def __iter__(self) -> Iterator:
        if self.policy in ('strict', 'repeat'):
            while True:
                yield from _exact(self.dataloader, self.target_samples)

        elif self.policy == 'supplement':
            while True:
                yield from _exact(chain(self.dataloader, self.supplement_dataloader), self.target_samples)

    def __len__(self):
        return self.target_samples


def _exact(iterator: Iterable[Batch], total: int) -> Generator[Batch, None, None]:
    returned = 0
    for (samples, labels) in iterator:
        take = min(samples.size(0), total - returned)
        yield samples[:take], labels[:take]

        returned += take
        if returned >= total:
            break
