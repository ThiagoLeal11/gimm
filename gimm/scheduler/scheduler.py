from abc import ABC, abstractmethod

import torch.optim


class SchedulerFunction(ABC):
    def __init__(self, initial_value: float, last_step: int = -1):
        self.initial_value = initial_value
        self.last_step = last_step

    @abstractmethod
    def __call__(self, t: int) -> float:
        pass


# TODO: adicionar qual t vai comecar, adicionar lr inicial.
class Scheduler(ABC):
    def __init__(
        self,
        param_name: str = "lr",
        last_step: int = -1,
        updates_per_step: int = 1,
    ):
        self.optimizer: torch.optim.Optimizer = None
        self.param_name = param_name
        self.initial_param_name = f"initial_{param_name}"
        self.updates_per_step = updates_per_step

        self.last_step = last_step
        self.current_step = 0

        self.base_lrs: list[float] = []
        self.current_loss = 0.0

    def construct(self, optimizer: torch.optim.Optimizer) -> 'Scheduler':
        self.optimizer = optimizer

        self.base_lrs = [
            group.get(self.initial_param_name) or group.get(self.param_name)
            for group in self.optimizer.param_groups
        ]
        return self

    def state_dict(self) -> dict[str, any]:
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict: dict[str, any]) -> None:
        self.__dict__.update(state_dict)

    @abstractmethod
    def _compute_lr(self, t: int) -> list[float]:
        pass

    def update_groups(self, values: list[float]):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)

        for param_group, value in zip(self.optimizer.param_groups, values):
            if "lr_scale" in param_group:
                param_group[self.param_name] = value * param_group["lr_scale"]
            else:
                param_group[self.param_name] = value

    def get_current_lrs(self) -> list[float]:
        return [group[self.param_name] for group in self.optimizer.param_groups]

    def compute_step(self, t: int) -> list[float]:
        if t is None:
            t = self.current_step + 1
            self.current_step = t

        return self._compute_lr(t)

    def step(self, t: int = None, current_loss: float = None) -> list[float]:
        self.current_loss = current_loss

        lrs = self.compute_step(t)
        self.update_groups(lrs)
        return lrs
