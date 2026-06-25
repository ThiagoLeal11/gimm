from gimm.scheduler.scheduler import Scheduler


class ConstantLR(Scheduler):
    def __init__(self, param_name: str = 'lr', updates_per_step: int = 1):
        super(ConstantLR, self).__init__(param_name, -1, updates_per_step)

    def compute_step(self, t: int) -> list[float]:
        return self.base_lrs

    def __repr__(self):
        return f"ConstantLR({self.base_lrs})"
