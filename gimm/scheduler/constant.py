import torch.optim

from gimm.scheduler.scheduler import Scheduler


class ConstantLR(Scheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, param_name: str = 'lr'):
        self.optimizer = optimizer
        super(ConstantLR, self).__init__(optimizer, param_name, -1)
        self.lr = [opt[self.param_name] for opt in self.optimizer.param_groups]

    def _compute_lr(self, t: int) -> list[float]:
        return self.lr
