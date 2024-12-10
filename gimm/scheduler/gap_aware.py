""" Gap-aware Learning Rate Scheduler for Adversarial Networks

A dynamic scheduler for discriminator learning rates based on the ideal gap loss between the adversarial networks.

Implementation adapted from:
    https://github.com/google-research/google-research/tree/master/adversarial_nets_lr_scheduler

Papers:
    Mind the (optimality) Gap: A Gap-Aware Learning Rate Scheduler for Adversarial Nets - https://arxiv.org/abs/2302.00089
"""


from math import log

from gimm.scheduler.scheduler import Scheduler


class GapAwareLR(Scheduler):
    """Gap-aware Learning Rate Scheduler for Adversarial Networks.

      The scheduler changes the learning rate of the discriminator (D) during
      training in an attempt to keep D's current loss close to that of D's ideal
      loss, i.e., D's loss when the distribution of the generated data matches
      that of the real data. The scheduler is called at every training step.
      See the paper for more details.

      Let x := abs(loss-ideal_loss) be the optimality gap. The scheduling function
      s(x) is a piecewise function defined as follows:
          f(x) := min(f_max^(x/x_max), f_max)  if loss >= ideal_loss
          h(x) := max(h_min^(x/x_min), h_min)  if loss < ideal_loss.

      When loss >= ideal_loss, the scheduler increases the learning rate in
      proportion to the optimality gap x. At x=x_max, the scheduler reaches its
      maximum value f_max. Similarly, when loss < ideal_loss, the scheduler
      decreases the learning rate. At x=x_min, it reaches the minimum allowed
      value h_min.

      Note: This function outputs s(x), which can be used to multiply the (base)
      learning rate of the optimizer.

      self.smoothed_loss: the loss of the discriminator D on the training data. In the paper,
      we estimate this quantity by using an exponential moving average of
      the loss of D over all batches seen so far (see the section "Example
      Usage for Training a GAN" in this colab for an example of the
      exponential moving average).

      Args:
        ideal_loss: the ideal loss of D. See Table 1 in our paper for the
          ideal loss of common GAN loss functions.
        x_min: the value of x at which the scheduler achieves its minimum allowed
          value h_min. Specifically, when loss < ideal_loss, the scheduler
          gradually decreases the LR (as long as x < x_min). For x >= x_min, the
          scheduler's output is capped to the minimum allowed value h_min. In the
          paper we set this to 0.1*ideal_loss.
        x_max: the value of x at which the scheduler achieves its maximum allowed
          value f_max. Specifically, when loss >= ideal_loss, the scheduler
          gradually increases the LR (as long as x < x_max). For x >= x_max, the
          scheduler's output is capped to the maximum allowed value f_max. In the
          paper we set this to 0.1*ideal_loss.
        h_min: a scalar in (0, 1] denoting the minimum allowed value of the
          scheduling function. In the paper we used h_min=0.1.
        f_max: a scalar (>= 1) denoting the maximum allowed value of the
          scheduling function. In the paper we used h_max=2.0.

      Returns:
        A scalar in [h_min, f_max], which can be used as a multiplier for the
          learning rate of D.
    """

    def __init__(
        self,
        optimizer,
        param_name: str = "lr",
        last_step: int = -1,
        ideal_loss: float = log(4),
        x_min: float = 0.1,
        x_max: float = 0.1,
        h_min: float = 0.1,
        f_max: float = 2.0,
    ):
        super(GapAwareLR, self).__init__(optimizer, param_name, last_step)

        self.ideal_loss = ideal_loss
        self.x_min = x_min
        self.x_max = x_max
        self.h_min = h_min
        self.f_max = f_max
        self.current_loss = 0.0

        self.smoothed_loss = None

    def _compute_lr(self, t: int) -> list[float]:
        x = abs(self.current_loss - self.ideal_loss)
        f_x = clip(self.f_max ** (x / self.x_max), 1.0, self.f_max)
        h_x = clip(self.h_min ** (x / self.x_min), self.h_min, 1.0)

        factor = f_x if self.current_loss > self.ideal_loss else h_x

        return [
            lr * factor
            for lr in self.base_lrs
        ]

    def step(self, t: int = None, current_loss: float = None) -> list[float]:
        if self.smoothed_loss is None:
            self.smoothed_loss = current_loss

        self.smoothed_loss = 0.95 * self.smoothed_loss + 0.05 * current_loss
        return super().step(t, current_loss)


def clip(x: float, lower: float, upper: float) -> float:
    return max(lower, min(x, upper))
