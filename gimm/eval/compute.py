from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from gimm.models.definition import ModuleGAN


# TODO: Precisa contar a distribuição real e a fake
# TODO: É legal cachear a real para evitar retrabalho desnecessário
# TODO: É legal ter uma forma de cálculo interativo (generator), pra evitar ficar regerando sempre a mesma coisa
# TODO: Definir todas as métricas na criação do training
# TODO: Permitir que as métricas sejam aplicadas de forma individual


class EvalMetric(ABC):
    def __init__(self, samples: int, device: torch.device = None):
        self.samples = samples
        self.device = device or torch.device('cpu')
        self.real_dist = {}

    def should_compute_real_distribution(self) -> bool:
        return not self.real_dist

    def reset(self) -> None:
        self.reset_real_distribution()
        self.reset_fake_distribution()

    @abstractmethod
    def reset_real_distribution(self) -> None:
        pass

    @abstractmethod
    def reset_fake_distribution(self) -> None:
        pass

    @abstractmethod
    def update(self, batch: tuple[Tensor, Tensor]) -> None:
        pass

    @abstractmethod
    def compute(self) -> dict[str, any]:
        pass


def compute_metrics(model: ModuleGAN, dataloader: DataLoader, metrics: list[EvalMetric]) -> dict[str, any]:
    # TODO: For each batch of real data, compute the real metric for every metric
    # TODO: For max number of samples, generate batch of fake data and compute the fake metric for every metric
    # TODO: Return the results
    pass