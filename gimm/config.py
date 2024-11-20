import os
from dataclasses import dataclass


@dataclass
class Config:

    data_dir: str = os.environ.get("PATH_DATASETS", ".")
    batch_size: int = 64
    num_workers = 4


@dataclass
class ModelConfig:
    # Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty
    input_size: list[int] = None

    # # Name of model to train
    # model: str = "resnet50"
    # # Start with pretrained version of specified network (if avail)
    # pretrained: bool = False
    # # Load this checkpoint as if they were the pretrained weights (with adaptation).
    # pretrained_path: str = None
    # # Load this checkpoint into model after initialization (default: none)
    # initial_checkpoint: str = ""
    # # Resume full model and optimizer state from checkpoint (default: none)
    # resume: str = ""
    # # prevent resume of optimizer state when resuming model
    # no_resume_opt: bool = False
    # # number of label classes (Model default if None)
    # num_classes: int = None
    # # Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.
    # gp: str = None
    # # Input image center crop percent (for validation only)
    # crop_pct: float = None
    # # Override mean pixel value of dataset
    # mean: list[float] = None
    # # Override std deviation of dataset
    # std: list[float] = None
    # # Image resize interpolation type (overrides model)
    # interpolation: str = ""
    # # Validation batch size override (default: None)
    # validation_batch_size: int = None
    # # Use channels_last memory layout
    # channels_last: bool = False
    # Enable gradient checkpointing through model blocks/stages
    # grad_checkpointing: bool = False
    # # enable experimental fast-norm
    # fast_norm: bool = False
    # # Model default if None.
    # model_kwargs: dict = {}
    # # Head initialization scale
    # head_init_scale: float = None
    # # Head initialization bias value
    # head_init_bias: float = None


@dataclass
class Train:
    # Input batch size for training (default: 128)
    batch_size: int = 128
    # The number of steps to accumulate gradients (default: 1)
    grad_accum_steps: int = 1




# @dataclass
# class ModelOptimization:
    # # Select jit fuser. One of ('', 'te', 'old', 'nvfuser')
    # fuser: str = ""
    # # torch.compile mode (default: None).
    # torchcompile_mode: str = None
    # # torch.jit.script the full model
    # torchscript: bool = False
    # # Enable compilation w/ specified backend (default: inductor).
    # torchcompile: str = None


@dataclass
class Device:
    # Device (accelerator) to use.
    device: str = "cuda"

    # # use NVIDIA Apex AMP or Native AMP for mixed precision training
    # amp: bool = False
    # # lower precision AMP dtype (default: float16)
    # amp_dtype: str = "float16"
    # # AMP impl to use, "native" or "apex" (default: native)
    # amp_impl: str = "native"
    # # Force broadcast buffers for native DDP to off.
    # no_ddp_bb: bool = False
    # # torch.cuda.synchronize() end of each step
    # synchronize_step: bool = False
    # # Local rank of the process
    # local_rank: int = 0
    # # Python imports for device backend modules.
    # device_modules: list[str] = None


@dataclass
class OptimizerConfig:
    # Optimizer (default: "sgd")
    opt: str = "sgd"
    # # Optimizer Epsilon (default: None, use opt default)
    # opt_eps: float = None
    # # Optimizer Betas (default: None, use opt default)
    # opt_betas: list[float] = None
    # # Optimizer momentum (default: 0.9)
    # momentum: float = 0.9
    # # weight decay (default: 2e-5)
    # weight_decay: float = 2e-5
    # # Clip gradient norm (default: None, no clipping)
    # clip_grad: float = None
    # # Gradient clipping mode. One of ("norm", "value", "agc")
    # clip_mode: str = "norm"
    # # layer-wise learning rate decay (default: None)
    # layer_decay: float = None
    # # Optimizer kwargs
    # opt_kwargs: dict = {}


@dataclass
class LearningRateConfig:
    # LR scheduler (default: "cosine"
    scheduler: str = "cosine"
    # learning rate, overrides lr-base if set (default: None)
    lr: float = None
    # lower lr bound for cyclic schedulers that hit 0 (default: 0)
    min_lr: float = 0
    # number of epochs to train (default: 300)
    epochs: int = 300

    # warmup learning rate (default: 1e-5)
    warmup_lr: float = 1e-5
    # epochs to warmup LR, if scheduler supports
    warmup_epochs: int = 5
    # # manual epoch number (useful on restarts)
    # start_epoch: int = None
    # # list of decay epoch indices for multistep lr. must be increasing
    # decay_milestones: list[int] = [90, 180, 270]
    # # epoch interval to decay LR
    # decay_epochs: float = 90
    # # Exclude warmup period from decay schedule.
    # warmup_prefix: bool = False
    # # epochs to cooldown LR at min_lr, after cyclic schedule ends
    # cooldown_epochs: int = 0
    # # patience epochs for Plateau LR scheduler (default: 10)
    # patience_epochs: int = 10
    # # LR decay rate (default: 0.1)
    # decay_rate: float = 0.1

@dataclass
class ExponentialMovingAverage:
    # Enable Exponential Moving Average
    use_ema: bool = False
    # EMA decay rate
    ema_decay: float = 0.9999
    # EMA start epoch
    ema_start: int = 0
    # EMA update frequency
    ema_freq: int = 1
    # EMA device
    ema_force_cpu: bool = False


@dataclass
class Misc:
    # Random seed (default: 42)
    seed: int = 42
    # how many batches to wait before logging training status
    log_interval: int = 10
    # how many training processes to use (default: 4)
    num_workers: int = 4
    # path to output folder (default: none, current dir)
    output_dir: str = ""
    # Log into weights and biases
    wandb: bool = False
