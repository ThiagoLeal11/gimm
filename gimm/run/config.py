from dataclasses import dataclass, field
from typing import Optional, Literal

import torch

from gimm.eval.fidelity import MetricName
from gimm.run.optimizer import Optimizer
from gimm.scheduler.scheduler import Scheduler


@dataclass
class StopCondition:
    metric: str
    comparator: str
    threshold: float
    _valid_comparators = ['<', '>', '<=', '>=']

    def __post_init__(self):
        if self.comparator not in self._valid_comparators:
            raise ValueError(f"Invalid comparator: {self.comparator}. Must be one of {self._valid_comparators}")

        self.metric = self.metric.lower()
        valid_metrics = [v[1] for v in vars(MetricName).values() if isinstance(v, tuple) and len(v) == 2]
        if self.metric not in valid_metrics:
            raise ValueError(f"Invalid metric name: {self.metric}. Valid options: {sorted(list(valid_metrics))}")

    def check(self, value: float) -> bool:
        if self.comparator == '<':
            return value < self.threshold
        elif self.comparator == '>':
            return value > self.threshold
        elif self.comparator == '<=':
            return value <= self.threshold
        elif self.comparator == '>=':
            return value >= self.threshold
        return False

    def __str__(self):
        return f"{self.metric.upper()} {self.comparator} {self.threshold}"

    @classmethod
    def from_exp(cls, expression: str) -> 'StopCondition':
        for op in cls._valid_comparators:
            expression = expression.replace(op, f' {op} ')
        parts = expression.split()
        if len(parts) != 3:
            raise ValueError(f"Invalid expression format: '{expression}'. Expected format: 'metric comparator value' (e.g. 'fid < 10.5')")
        return cls(metric=parts[0], comparator=parts[1], threshold=float(parts[2]))


@dataclass
class TrainerConfig:
    project: str = 'gimm'
    group: Optional[str] = None
    execution_name: Optional[str] = None
    num_workers: int = 6

    batch_size: int = 128
    # Split batch into sub-batches due to GPU memory limitations
    grad_accum_steps: int = 1

    # Number of updates per step for discriminator and generator.
    # e.g. 5 means perform 5 minibatch updates grouped as a single step.
    d_updates_per_step: int = 1
    g_updates_per_step: int = 1

    # How many steps to increment after each batch
    steps_per_batch: int = 1

    # The number of batches used in the training loop
    steps_total: int = 200_000

    # Controlling
    steps_checkpoint: int = 10000
    steps_eval: int = 10000
    steps_log: int = 100
    steps_image: int = 1000

    # Number of images to log every steps_per_image
    images_to_log: int = 128

    # Optimizers
    g_optimizer: Optimizer = None
    d_optimizer: Optimizer = None

    # Schedulers
    g_scheduler: Scheduler = None
    d_scheduler: Scheduler = None

    # Checkpoint
    output_path: str = "output"
    resume_checkpoint: str = ''  # 'output/checkpoints/checkpoint-150_000.pth.tar'
    delete_checkpoint: bool = False
    checkpoint_resume_config: bool = True
    checkpoint_weights_only: bool = False

    # Useful for distribute default configs for the model training
    dataset: Optional[str] = None
    dataset_dir: Optional[str] = '.'
    # 'auto': warns on rescaling; 'strict': error if resolution differs; 'silent': rescale without warning.
    dataset_scale_policy: Literal['auto', 'strict', 'silent'] = 'auto'
    dataset_bake: bool = False
    dataset_bake_type: Literal['memory', 'lmdb', 'folder'] = 'memory'
    dataset_bake_path: Optional[str] = 'output/baked_dataset/'

    device: Optional[torch.device] = None

    stop_conditions: list[StopCondition] = field(default_factory=list)

    # Validation
    validation_samples: int = 20_000
    validation_policy: Literal['repeat', 'supplement', 'strict'] = 'strict'
    split_config: Optional[list[int]] = None

    def __post_init__(self):
        assert  self.steps_per_batch >= 1, "steps_per_batch must be at least 1."
        assert self.grad_accum_steps > 0, "grad_accum_steps must be at least 1."
        assert self.d_updates_per_step >= 1, "d_updates_per_step must be at least 1."
        assert self.g_updates_per_step >= 1, "g_updates_per_step must be at least 1."

        # Assert controlling steps are all multiples of steps_per_batch
        assert self.steps_checkpoint % self.steps_per_batch == 0, "steps_checkpoint must be a multiple of steps_per_batch."
        assert self.steps_eval % self.steps_per_batch == 0, "steps_eval must be a multiple of steps_per_batch."
        assert self.steps_log % self.steps_per_batch == 0, "steps_log must be a multiple of steps_per_batch."
        assert self.steps_image % self.steps_per_batch == 0, "steps_image must be a multiple of steps_per_batch."
        assert self.steps_total % self.steps_per_batch == 0, "steps_total must be a multiple of steps_per_batch."
