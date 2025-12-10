from gimm.scheduler.scheduler import Scheduler


class ExponentialLR(Scheduler):
    def __init__(self, last_step: int, gamma: float, param_name: str = "lr", updates_per_step: int = 1):
        """
        Exponential learning rate scheduler.
        
        Decays the learning rate exponentially from base_lr to base_lr * gamma over last_step steps.
        lr_t = base_lr * gamma^(t / last_step)

        At t=0: lr = base_lr
        At t=last_step: lr = base_lr * gamma

        Args:
            last_step: Total number of steps.
            gamma: Final multiplicative factor of learning rate decay (e.g., 0.01 means lr decays to 1% of initial).
            param_name: Name of the parameter to schedule.
            updates_per_step: Number of updates per step.
        """
        super(ExponentialLR, self).__init__(param_name, last_step, updates_per_step)
        self.gamma = gamma

    def _compute_lr(self, t: int) -> list[float]:
        # t starts at 1 after first step() call, so we use (t-1) to start from base_lr
        progress = (t - 1) / self.last_step if self.last_step > 0 else 0
        return [base_lr * (self.gamma ** progress) for base_lr in self.base_lrs]

    def __repr__(self):
        return f"ExponentialLR(gamma={self.gamma}, last_step={self.last_step}, base_lrs={self.base_lrs})"

