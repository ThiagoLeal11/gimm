import torch.optim

from gimm.scheduler.scheduler import Scheduler


class ConstantLR(Scheduler):
    def __init__(self, param_name: str = 'lr', updates_per_step: int = 1):
        super(ConstantLR, self).__init__(param_name, -1, updates_per_step)
        self.lr = []

    def construct(self, optimizer: torch.optim.Optimizer) -> 'ConstantLR':
        super(ConstantLR, self).construct(optimizer)
        self.lr = [opt[self.param_name] for opt in self.optimizer.param_groups]
        return self

    def _compute_lr(self, t: int) -> list[float]:
        return self.lr
