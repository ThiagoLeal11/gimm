from gimm.scheduler.scheduler import Scheduler
from gimm.scheduler.tween import Tween


class CombineSchedulers(Scheduler):
    def __init__(
        self,
        schedulers: list[Scheduler],
        tween: Tween,
        param_name: str = "lr",
        last_step: int = -1,
    ):
        optimizer = schedulers[0].optimizer
        super().__init__(optimizer, param_name, last_step)

        self.schedulers = schedulers
        self.tween = tween

        if len(schedulers) != 2:
            raise ValueError("CombineSchedulers only supports two schedulers.")

    def _compute_lr(self, t: int) -> list[float]:
        tween = self.tween(t)
        values = [
            scheduler.compute_step(t) for scheduler in self.schedulers
        ]
        return [a * (1 - tween) + b * tween for a, b in zip(values[0], values[1])]
