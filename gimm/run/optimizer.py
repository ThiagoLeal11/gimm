from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Tuple

import torch.optim
from torch.optim.optimizer import ParamsT


@dataclass
class Optimizer(ABC):
    name: str = 'None'

    @abstractmethod
    def construct(self, params: ParamsT) -> torch.optim.Optimizer:
        """
        Constructs the optimizer with the given parameters.
        """
        raise NotImplementedError("Optimizer must implement the construct method.")


@dataclass
class Adam(Optimizer):
    name: str = 'Adam'
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0.0
    ams_grad: bool = False
    fused: bool = False

    def construct(self, params: ParamsT) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.ams_grad,
            fused=self.fused
        )

    def __dict__(self):
        return {
            'name': self.name,
            'lr': self.lr,
            'betas': self.betas,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'ams_grad': self.ams_grad,
            'fused': self.fused
        }
