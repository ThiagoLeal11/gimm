import math

from gimm.scheduler.scheduler import Scheduler


class CosineLR(Scheduler):
    def _compute_lr(self, t: int) -> list[float]:
        lrs = []
        for base_lr in self.base_lrs:
            lrs.append(base_lr * 0.5 * (1 + math.cos(math.pi * t / self.last_step)))
        return lrs
