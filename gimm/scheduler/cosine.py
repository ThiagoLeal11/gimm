import math

from gimm.scheduler.scheduler import Scheduler


class CosineLR(Scheduler):
    def __init__(self, last_step: int, param_name: str = "lr", updates_per_step: int = 1):
        super(CosineLR, self).__init__(param_name, last_step, updates_per_step)

    def _compute_lr(self, t: int) -> list[float]:
        lrs = []
        for base_lr in self.base_lrs:
            lrs.append(base_lr * 0.5 * (1 + math.cos(math.pi * (t-1) / self.last_step)))
        return lrs

    def __repr__(self):
        return f"CosineLR(lr={self.base_lrs[0]}, end_step={self.last_step})"