import pathlib
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Literal, Sequence

import torch
from torchvision import transforms

from gimm.chekpoint import Checkpoint, clean_dir_deep
from gimm.datasets.definition import Dataset
from gimm.dataloaders.infinite import InfinitePrefetchLoader
from gimm.dataloaders.eval import ValidationLoader
from gimm.eval.fidelity import FidelityEvalMetric, compute_metrics
from gimm.logs.log import Logger, get_wandb_run_id, LoggerWandb
from gimm.models.definition import ModuleGAN
from gimm.run.loader import dataset_loader
from gimm.run.optimizer import Optimizer
from gimm.scheduler.scheduler import Scheduler
from gimm.eval.fidelity import MetricName


def custom_format_warning(message, category, filename, lineno, line=None):
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"
warnings.formatwarning = custom_format_warning


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

    # Useful for distribute default configs for the model training
    dataset: Optional[str] = None
    dataset_dir: Optional[str] = '.'
    # 'auto': warns on rescaling; 'strict': error if resolution differs; 'silent': rescale without warning.
    dataset_scale_policy: Literal['auto', 'strict', 'silent'] = 'auto'

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


class Trainer:
    def __init__(self, model: ModuleGAN, configs: TrainerConfig, loggers: list[Logger], eval_metrics: list[FidelityEvalMetric]):
        self.configs = configs
        self.loggers = loggers
        self.step = 0

        self.model = model
        self.device = configs.device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fixed_z = self.model.get_latent(configs.images_to_log)

        # Define the optimizers and schedulers
        self.optimizer_g = self.configs.g_optimizer.construct(self.model.generator.parameters())
        self.optimizer_d = self.configs.d_optimizer.construct(self.model.discriminator.parameters())
        self.lr_scheduler_g = self.configs.g_scheduler.construct(optimizer=self.optimizer_g)
        self.lr_scheduler_d = self.configs.d_scheduler.construct(optimizer=self.optimizer_d)

        self.checkpoint = Checkpoint(
            configs=configs.__dict__,
            args={
                'wandb_run_id': get_wandb_run_id(loggers),
                'fixed_z': self.fixed_z,
            },
            model=self.model,
            optimizer_generator=self.optimizer_g,
            optimizer_discriminator=self.optimizer_d,
            checkpoint_dir=configs.output_path + '/checkpoints/',
            raise_if_dir_not_empty=not configs.resume_checkpoint,
            clean_checkpoint_dir=configs.delete_checkpoint,
        )

        if configs.resume_checkpoint:
            print('resuming_checkpoint')
            self.resume_checkpoint(self.configs.resume_checkpoint)

            args = self.checkpoint.args
            self.fixed_z = args['fixed_z']

            # Set the wandb resume_id if a wandb logger is present
            wandb_logger = next((logger for logger in self.loggers if isinstance(logger, LoggerWandb)), None)
            if wandb_logger:
                wandb_logger.set_resume(args['wandb_run_id'])

        # Finished starting loggers
        for logger in self.loggers:
            logger.start()

        self.eval_metrics = eval_metrics

        # Cleans evaluation cache from other runs
        clean_dir_deep(pathlib.Path(self.configs.output_path) / 'cache' / 'fidelity_cache')

    def save_checkpoint(self):
        self.checkpoint.save(step=self.step)

    def resume_checkpoint(self, path: str):
        self.step = self.checkpoint.load(path, should_resume_config=True)

    def before_training(self, data: Dataset):
        self.model.to(self.device)

    def before_evaluate(self, data: Dataset):
        self.model.to(self.device)

    def after_evaluate(self):
        pass

    def after_training(self):
        self.save_checkpoint()

    @staticmethod
    def scale_dataset_to_model(dataset: Dataset, model_in_features: Sequence[int], scale_policy: Literal['auto', 'strict', 'silent'] = 'auto') -> Dataset:
        orig_ch, orig_h, orig_w = dataset.dims
        target_ch, target_h, target_w = model_in_features

        if orig_ch != target_ch:
            raise ValueError(f"Dataset channels ({orig_ch}) do not match model input features ({target_ch}).")

        is_same_resolution = orig_h == target_h and orig_w == target_w
        if is_same_resolution:
            return dataset

        if scale_policy == 'strict':
            raise ValueError(f"Dataset dimensions ({orig_h}x{orig_w}) do not match model input features ({target_h}x{target_w}).")
        elif scale_policy == 'auto':
            message = f"Rescaling dataset from ({orig_h}x{orig_w}) to ({target_h}x{target_w})."
            warnings.warn(message)

        scale = max(target_h / orig_h, target_w / orig_w)
        new_size = (int(orig_h * scale), int(orig_w * scale))

        dataset.update_transformations([
            transforms.Resize(new_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop((target_h, target_w))
        ])
        return dataset

    def get_image_examples(self):
        bs = self.configs.batch_size
        zs = self.fixed_z.to(self.device)
        generated_images: list[torch.Tensor] = []

        with torch.no_grad():
            for i in range(0, zs.size(0), bs):
                z_batch = zs[i:i + bs]
                imgs = self.model(z_batch)
                generated_images.append(imgs)

        return torch.cat(generated_images, dim=0)

    def load_dataset(self, data: Optional[Dataset] = None):
        if data is not None:
            assert isinstance(data, Dataset), f"Expected data to be a Dataset instance, got {type(data)}"
            return data

        cfg = self.configs
        assert cfg.dataset, f"Dataset name must be provided in configs, got {cfg.dataset}"

        dataset = dataset_loader(
            name=cfg.dataset,
            batch_size=cfg.batch_size * cfg.grad_accum_steps,
            num_workers=cfg.num_workers,
            data_dir=cfg.dataset_dir,
            split_config=cfg.split_config,
        )

        if self.configs.validation_policy == 'strict' and dataset.get_splits()[1] < cfg.validation_samples:
            raise ValueError(f"Validation set size ({dataset.get_splits()[1]}) is smaller than validation_samples ({cfg.validation_samples}) under 'strict' policy. Change to 'repeat', 'supplement' (with complete the samples with train samples), or reduce validation_samples.")

        return self.scale_dataset_to_model(dataset, self.model.in_features, cfg.dataset_scale_policy)

    def train(self, data: Optional[Dataset] = None):
        data = self.load_dataset(data)
        try:
            self.before_training(data)
            self.model.train()

            self.optimizer_g.zero_grad()
            self.optimizer_d.zero_grad()

            self.train_loop(data)

            self.after_training()
        finally:
            self.finish_loggers()

    def train_loop(self, data: Optional[Dataset] = None):
        step = self.step
        total_inner_steps = max(self.configs.g_updates_per_step, self.configs.d_updates_per_step)
        inner_step = total_inner_steps

        train_dataloader = data.train_dataloader()
        sampler = InfinitePrefetchLoader(train_dataloader, device=self.device, infinite=True, preload=True)

        for (imgs, labels) in sampler:
            metrics = {}
            for accum_idx in range(self.configs.grad_accum_steps):
                start_idx = accum_idx * self.configs.batch_size
                end_idx = start_idx + self.configs.batch_size

                # Execute the training step
                batch = (imgs[start_idx:end_idx], labels[start_idx:end_idx])
                metrics = self.training_step(
                    global_step=step,
                    inner_step=inner_step,
                    accum_idx=accum_idx,
                    batch=batch
                )

            # Ensure every inner step is executed
            inner_step -= 1
            if inner_step > 0:
                continue
            inner_step = total_inner_steps

            if step % self.configs.steps_log == 0:
                for logger in self.loggers:
                    train_metrics = {f'train_{k}': v for k, v in metrics.items()}
                    logger.log(step=step, metrics=train_metrics)

            if step % self.configs.steps_image == 0 and step > 0:
                generated_image_examples = self.get_image_examples()
                for logger in self.loggers:
                    logger.log_image(step=step, image=generated_image_examples, prefix='train')

            if step == 0:
                for logger in self.loggers:
                    real_images = imgs[:self.configs.images_to_log]
                    logger.log_image(step=step, image=real_images, prefix='real')

            if step % self.configs.steps_checkpoint == 0 and step > 0:
                print('Salvando checkpoint', self.step)
                self.save_checkpoint()

            if step % self.configs.steps_eval == 0 and step > 0:
                self.model.eval()
                metrics = self.evaluate(data)
                self.model.train()

                for logger in self.loggers:
                    eval_metrics = {f'eval_{k}': v for k, v in metrics.items()}
                    logger.log(step=step, metrics=eval_metrics)

                # Check for early stopping
                should_stop = False
                for condition in self.configs.stop_conditions:                    
                    if condition.metric not in metrics:
                        #TODO: adicionar validação de compilação (quais métricas estão ativas para quais ele está configurando no stop_condition)
                        continue

                    if condition.check(metrics[condition.metric]):
                        print(f"\n[Early Stopping] Condition met: {condition} (Current value: {metrics[condition.metric]:.4f})")
                        should_stop = True

                if should_stop:
                    break

            if step >= self.configs.steps_total:
                break

            # All logs are taken from the start step.
            self.step += self.configs.steps_per_batch
            step = self.step

    def training_step(self, global_step: int, inner_step: int, accum_idx: int, batch: tuple[torch.Tensor, torch.Tensor]) -> dict:
        is_last_accum_batch = (accum_idx + 1) % self.configs.grad_accum_steps == 0
        imgs, labels = batch

        self.model.current_step = global_step

        time_start = time.time()

        # train discriminator
        d_loss = None
        if inner_step <= self.configs.d_updates_per_step:
            fake_imgs = self.model.get_previous_generated_samples(imgs)
            d_loss = self.model.compute_discriminator_loss(imgs, fake_imgs)
            self.backward('d', d_loss, should_step=is_last_accum_batch)

        # train generator
        g_loss = None
        if inner_step <= self.configs.g_updates_per_step:
            g_loss = self.model.compute_generator_loss(imgs)
            self.backward('g', g_loss, should_step=is_last_accum_batch)

        time_end = time.time()

        lr_g = self.optimizer_g.param_groups[0]['lr']
        lr_d = self.optimizer_d.param_groups[0]['lr']
        it_s = imgs.size(0) / (time_end - time_start)
        metrics = {
            'g_loss': compute_item(g_loss), 'd_loss': compute_item(d_loss), 'lr_g': lr_g, 'lr_d': lr_d, 'its': it_s
        }
        return metrics

    def backward(self, module: Literal['g', 'd'], loss: Sequence[torch.Tensor] | torch.Tensor, should_step: bool):
        accum_steps = self.configs.grad_accum_steps
        if accum_steps > 1:
            loss /= accum_steps

        if isinstance(loss, torch.Tensor):
            loss.backward()
        else:
            torch.autograd.backward(loss)

        if should_step:
            if module == 'g':
                self.model.before_generator_step()
                optimizer = self.optimizer_g
            else:
                self.model.before_discriminator_step()
                optimizer = self.optimizer_d

            optimizer.step()
            optimizer.zero_grad()

            if module == 'g':
                self.model.after_generator_step()
                lr_scheduler = self.lr_scheduler_g
            else:
                self.model.after_discriminator_step()
                lr_scheduler = self.lr_scheduler_d

            lr_scheduler.step(current_loss=compute_item(loss))


    def evaluate(self, data: Optional[Dataset] = None) -> dict:
        data = self.load_dataset(data)

        self.before_evaluate(data)
        
        # Configure ValSampler
        val_loader = data.validation_dataloader()
        train_loader = None
        if self.configs.validation_policy == 'supplement':
            train_loader = data.train_dataloader()

        sampler = ValidationLoader(
            dataloader=val_loader,
            supplement_dataloader=train_loader,
            target_samples=self.configs.validation_samples,
            policy=self.configs.validation_policy
        )
        
        self.model.eval()

        if self.device.type.startswith('cuda') and not self.device.type.endswith(':0'):
            torch.cuda.set_device(self.device)

        metrics_value = compute_metrics(model=self.model, dataloader=sampler, metrics=self.eval_metrics, config={
            'output_path': self.configs.output_path,
            'experiment': self.configs.project,
            'samples': self.configs.validation_samples,
            'batch_size': self.configs.batch_size,
            'device': self.device,
            'verbose': True,
        })

        return metrics_value

    def finish_loggers(self):
        for logger in self.loggers:
            logger.finish()


def compute_item(tensors: Sequence[torch.Tensor] | torch.Tensor) -> int | float | bool | None:
    if tensors is None:
        return None
    if isinstance(tensors, torch.Tensor):
        return tensors.item()
    return sum(t.item() for t in tensors)
