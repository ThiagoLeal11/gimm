import dataclasses
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

    def state_dict(self) -> dict:
        return dataclasses.asdict(self)

    def load_state_dict(self, state_dict: dict) -> None:
        for key, value in state_dict.items():
            setattr(self, key, value)


@dataclass
class Adam(Optimizer):
    name: str = 'Adam'
    lr: float = 1e-3
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


@dataclass
class SGD(Optimizer):
    name: str = 'SGD'
    lr: float = 1e-3
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False
    maximize: bool = False
    foreach: bool = None
    differentiable: bool = False
    fused: bool = None

    def construct(self, params: ParamsT) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            params,
            lr=self.lr,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
            maximize=self.maximize,
            foreach=self.foreach,
            differentiable=self.differentiable,
            fused=self.fused
        )


@dataclass
class RMSprop(Optimizer):
    name: str = 'RMSprop'
    lr: float = 1e-2
    alpha: float = 0.99
    eps: float = 1e-8
    weight_decay: float = 0.0
    momentum: float = 0.0
    centered: bool = False
    maximize: bool = False
    foreach: bool = None
    differentiable: bool = False

    def construct(self, params: ParamsT) -> torch.optim.Optimizer:
        return torch.optim.RMSprop(
            params,
            lr=self.lr,
            alpha=self.alpha,
            eps=self.eps,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            centered=self.centered,
            maximize=self.maximize,
            foreach=self.foreach,
            differentiable=self.differentiable
        )
