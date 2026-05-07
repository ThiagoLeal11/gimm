import pathlib
import time
import warnings
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
from gimm.run.config import TrainerConfig
from gimm.run.loader import dataset_loader
from gimm.run.optimizer import Optimizer
from gimm.scheduler.scheduler import Scheduler


def custom_format_warning(message, category, filename, lineno, line=None):
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"
warnings.formatwarning = custom_format_warning


class Trainer:
    def __init__(self, model: ModuleGAN, configs: TrainerConfig, loggers: list[Logger], eval_metrics: list[FidelityEvalMetric]):
        self.configs = configs
        self.loggers = loggers
        self.step = 0

        self.model = model
        self.device = configs.device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Log device utilization
        device = self.get_runtime_device(self.model, self.device)
        if device.type != 'cpu':
            device_index = device.index if device.index is not None else torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device_index)
            print(f'Modelo executando com aceleração: ({device_name}):{device_index}')
        else:
            warnings.warn(f'Modelo executando sem aceleração! Dispositivo atual: {device}.')

        self.fixed_z = self.model.get_latent(configs.images_to_log)

        # Define the optimizers and schedulers
        self.optimizers_g = self.create_optimizers(self.configs.g_optimizer, self.model.generators)
        self.optimizers_d = self.create_optimizers(self.configs.d_optimizer, self.model.discriminators)

        self.lr_schedulers_g = self.create_schedulers(self.configs.g_scheduler, self.optimizers_g)
        self.lr_schedulers_d = self.create_schedulers(self.configs.d_scheduler, self.optimizers_d)

        self.checkpoint = Checkpoint(
            configs=configs.__dict__,
            args={
                'wandb_run_id': get_wandb_run_id(loggers),
                'fixed_z': self.fixed_z,
            },
            model=self.model,
            optimizer_generator=self.optimizers_g,
            optimizer_discriminator=self.optimizers_d,
            checkpoint_prefix=configs.output_path + '/checkpoints/checkpoint-',
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

    @staticmethod
    def get_runtime_device(model: torch.nn.Module, fallback: torch.device) -> torch.device:
        parameter = next(model.parameters(), None)
        if parameter is not None:
            return parameter.device

        buffer = next(model.buffers(), None)
        if buffer is not None:
            return buffer.device

        return fallback

    def save_checkpoint(self):
        self.checkpoint.save(step=self.step)

    def resume_checkpoint(self, path: str):
        self.step = self.checkpoint.load(
            checkpoint_path=path,
            should_resume_config=self.configs.checkpoint_resume_config,
            weights_only=self.configs.checkpoint_weights_only
        )
        if self.configs.checkpoint_resume_config:
            self.configs.__dict__.update(self.checkpoint.configs)

    @staticmethod
    def create_optimizers(config_opt: Optimizer | dict[str, Optimizer], networks: torch.nn.ModuleDict) -> dict[str, torch.optim.Optimizer]:
        # Single config for all networks
        if not isinstance(config_opt, dict):
            return {
                name: config_opt.construct(net.parameters())
                for name, net in networks.items()
            }

        # Create an optimizer for each network
        optimizers = {}
        for name, opt_conf in config_opt.items():
            if name in networks:
                optimizers[name] = opt_conf.construct(networks[name].parameters())
        return optimizers

    @staticmethod
    def create_schedulers(config_sch: Scheduler | dict[str, Scheduler], optimizers: dict[str, torch.optim.Optimizer]):
        # Single config for all networks
        if not isinstance(config_sch, dict):
            return {
                name: config_sch.construct(optimizer=opt)
                for name, opt in optimizers.items()
            }

        # Create a scheduler for each network
        schedulers = {}
        for name, sch_conf in config_sch.items():
            if name in optimizers:
                schedulers[name] = sch_conf.construct(optimizer=optimizers[name])
        return schedulers

    def before_training(self, data: Dataset):
        self.model.to(self.device)

    def after_training(self):
        self.save_checkpoint()

    def before_evaluate(self, data: Dataset):
        self.model.to(self.device)

    def after_evaluate(self):
        pass

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

        dataset.add_static_transforms([
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
            bake=cfg.dataset_bake,
            bake_type=cfg.dataset_bake_type,
            bake_path=cfg.dataset_bake_path,
            bake_min_size=cfg.dataset_bake_min_size,
        )

        if self.configs.validation_policy == 'strict' and dataset.get_splits()[1] < cfg.validation_samples:
            raise ValueError(f"Validation set size ({dataset.get_splits()[1]}) is smaller than validation_samples ({cfg.validation_samples}) under 'strict' policy. Change to 'repeat', 'supplement' (with complete the samples with train samples), or reduce validation_samples.")

        return self.scale_dataset_to_model(dataset, self.model.in_features, cfg.dataset_scale_policy)

    def train(self, data: Optional[Dataset] = None):
        data = self.load_dataset(data)
        try:
            self.before_training(data)
            self.model.train()

            for opt_g in self.optimizers_g.values():
                opt_g.zero_grad()
            for opt_d in self.optimizers_d.values():
                opt_d.zero_grad()

            self.train_loop(data)

            self.after_training()
        finally:
            self.finish_loggers()
            if data is not None and hasattr(data, 'shutdown'):
                data.shutdown()

    def train_loop(self, data: Optional[Dataset] = None):
        step = self.step
        total_inner_steps = max(self.configs.g_updates_per_step, self.configs.d_updates_per_step)
        inner_step = total_inner_steps

        train_dataloader = data.train_dataloader()
        sampler = InfinitePrefetchLoader(
            train_dataloader, device=self.device, infinite=True, preload=True, continuous_shuffle=True
        )

        for (imgs, labels) in sampler:
            metrics = {}
            for accum_idx in range(self.configs.grad_accum_steps):
                start_idx = accum_idx * self.configs.batch_size
                end_idx = start_idx + self.configs.batch_size

                # Execute the training step
                batch = (imgs[start_idx:end_idx], labels[start_idx:end_idx])
                metrics = self.training_step(
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

    def training_step(self, inner_step: int, accum_idx: int, batch: tuple[torch.Tensor, torch.Tensor]) -> dict:
        """
        :param inner_step: The current inner training step (when using multiple updates per step).
        :param accum_idx: The current accumulation index (for gradient accumulation).
        :param batch: A tuple containing the input images and their corresponding labels.
        :return: A dictionary containing the training metrics.
        """
        self.model.current_step = self.step
        time_start = time.time()

        # train discriminator
        d_loss = None
        if inner_step <= self.configs.d_updates_per_step:
            d_backward = lambda l: self.backward('d', l, accum_idx)
            d_loss = self.model.discriminator_train_step(batch, d_backward)

        # train generator
        g_loss = None
        if inner_step <= self.configs.g_updates_per_step:
            g_backward = lambda l: self.backward('g', l, accum_idx)
            g_loss = self.model.generator_train_step(batch, g_backward)

        time_end = time.time()

        lrs_g = {
            f'lr_g_{k}' if len(self.optimizers_g) > 1 else 'lr_g': v.param_groups[0]['lr']
            for k, v in self.optimizers_g.items()
        }
        lrs_d = {
            f'lr_d_{k}' if len(self.optimizers_d) > 1 else 'lr_d': v.param_groups[0]['lr']
            for k, v in self.optimizers_d.items()
        }
        it_s = batch[0].size(0) / (time_end - time_start)
        metrics = {
            **self.model.log_generator_loss(g_loss),
            **self.model.log_discriminator_loss(d_loss),
            **lrs_g,
            **lrs_d,
            'its': it_s
        }
        return metrics

    def backward(self, module: Literal['g', 'd'], loss: Sequence[torch.Tensor] | torch.Tensor, accum_idx: int = 0):
        accum_steps = self.configs.grad_accum_steps
        is_last_accum_batch = (accum_idx + 1) % self.configs.grad_accum_steps == 0

        if accum_steps > 1:
            loss /= accum_steps

        if isinstance(loss, torch.Tensor):
            loss.backward()
        else:
            torch.autograd.backward(loss)

        if is_last_accum_batch:
            if module == 'g':
                self.model.generator_optimizer_step(self.optimizers_g)
                lr_schedulers = self.lr_schedulers_g
            else:
                self.model.discriminator_optimizer_step(self.optimizers_d)
                lr_schedulers = self.lr_schedulers_d

            for lrs in lr_schedulers.values():
                lrs.step(t=self.step, current_loss=compute_item(loss))

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
