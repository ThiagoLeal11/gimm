import time
from dataclasses import dataclass
from typing import Optional, Generator

import torch

from gimm.chekpoint import Checkpoint
from gimm.datasets.definition import Dataset
from gimm.datasets.sampler import SmartSampler
from gimm.eval.fidelity import FidelityEvalMetric, compute_metrics
from gimm.logs.log import Logger, get_wandb_run_id, LoggerWandb
from gimm.models.definition import ModuleGAN
from gimm.run.loader import dataset_loader
from gimm.run.optimizer import Optimizer
from gimm.scheduler.scheduler import Scheduler


@dataclass
class TrainerConfig:
    project: str = 'gimm'
    num_workers: int = 6

    batch_size: int = 128
    grad_accum_steps: int = 1

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

    # Useful for distribute default configs for the model training
    dataset: Optional[str] = None
    dataset_dir: Optional[str] = '.'

    device: Optional[torch.device] = None

    def __post_init__(self):
        assert  self.steps_per_batch >= 1, "steps_per_batch must be at least 1."
        assert self.grad_accum_steps > 0, "grad_accum_steps must be at least 1."

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
        self.optimizer_g: torch.optim.Optimizer = None
        self.optimizer_d: torch.optim.Optimizer = None
        self.lr_scheduler_g: Scheduler = None
        self.lr_scheduler_d: Scheduler = None

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
            raise_if_dir_not_empty=not configs.resume_checkpoint
        )

        if configs.resume_checkpoint:
            print('resuming_checkpoint')
            self.resume_checkpoint(self.configs.resume_checkpoint)

            args = self.checkpoint.args
            self.fixed_z = args['fixed_z']

            # Set the wandb resume id if a wandb logger is present
            wandb_logger = next((logger for logger in self.loggers if isinstance(logger, LoggerWandb)), None)
            if wandb_logger:
                wandb_logger.set_resume(args['wandb_run_id'])

        # Finished starting loggers
        for logger in self.loggers:
            logger.start()

        self.eval_metrics = eval_metrics
        self.is_model_constructed = False

    def construct_model(self, data: Dataset):
        if self.is_model_constructed:
            return

        # Construct the model with the input dimensions given by the dataset
        self.model.construct(in_features=data.dims)
        self.optimizer_g = self.configs.g_optimizer.construct(self.model.generator.parameters())
        self.optimizer_d = self.configs.d_optimizer.construct(self.model.discriminator.parameters())
        self.lr_scheduler_g = self.configs.g_scheduler.construct(optimizer=self.optimizer_g)
        self.lr_scheduler_d = self.configs.d_scheduler.construct(optimizer=self.optimizer_d)
        # Update checkpoint with the new optimizers
        self.checkpoint.optimizer_generator = self.optimizer_g
        self.checkpoint.optimizer_discriminator = self.optimizer_d
        # Declare the model as constructed
        self.is_model_constructed = True

    def save_checkpoint(self):
        self.checkpoint.save(step=self.step)

    def resume_checkpoint(self, path: str):
        self.step = self.checkpoint.load(path, should_resume_config=True)

    def before_training(self, data: Dataset):
        self.construct_model(data)
        self.model.generator.to(self.device)
        self.model.discriminator.to(self.device)

    def before_evaluate(self, data: Dataset):
        self.construct_model(data)
        self.model.generator.to(self.device)
        self.model.discriminator.to(self.device)

    def after_evaluate(self):
        pass

    def after_training(self):
        self.save_checkpoint()

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

    @staticmethod
    def load_dataset(configs: TrainerConfig, data: Optional[Dataset] = None, name: str = 'training'):
        if data is None:
            if not configs.dataset:
                raise ValueError(f"No dataset provided for {name} and no dataset name in configs.")
            bs = configs.batch_size * configs.grad_accum_steps
            data = dataset_loader(configs.dataset, bs, configs.num_workers, configs.device, configs.dataset_dir)
        return data

    def train(self, data: Optional[Dataset] = None):
        data = self.load_dataset(self.configs, data, 'training')
        self.before_training(data)
        self.model.train()

        self.optimizer_g.zero_grad()
        self.optimizer_d.zero_grad()

        step = self.step
        train_dataloader = data.train_dataloader()

        sampler = SmartSampler(train_dataloader, device=self.device, infinite=True, preload=True)
        for (imgs, labels) in sampler:
            metrics = {}
            for accum_idx in range(self.configs.grad_accum_steps):
                start_idx = accum_idx * self.configs.batch_size
                end_idx = start_idx + self.configs.batch_size

                # Execute the training step
                batch = (imgs[start_idx:end_idx], labels[start_idx:end_idx])
                metrics = self.training_step(accum_idx, batch)

            if step % self.configs.steps_log == 0:
                for logger in self.loggers:
                    train_metrics = {f'train_{k}': v for k, v in metrics.items()}
                    logger.log(step=step, metrics=train_metrics)

            if step % self.configs.steps_image == 0 and step > 0:
                generated_image_examples = self.get_image_examples()
                for logger in self.loggers:
                    logger.log_image(step=step, image=generated_image_examples, prefix='train')

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

            if step >= self.configs.steps_total:
                break

            # All logs are taken from the start step.
            self.step += self.configs.steps_per_batch
            step = self.step

        self.after_training()

    def training_step(self, accum_idx: int, batch: tuple[torch.Tensor, torch.Tensor]) -> dict:
        is_last_accum_batch = (accum_idx + 1) % self.configs.grad_accum_steps == 0
        imgs, labels = batch

        time_start = time.time()

        # TODO: implement updates_per_step

        # train generator
        # fake_imgs = None
        # if accum_idx < accum_steps * self.configs.g_updates_per_step:
        g_loss, fake_imgs = self.model.compute_generator_loss(imgs)
        self.backward(g_loss, self.lr_scheduler_g, self.optimizer_g, should_step=is_last_accum_batch)

        # train discriminator
        # if accum_idx < accum_steps * self.configs.d_updates_per_step:
        #     if not fake_imgs:
        #         fake_imgs = self.model.generate_images(imgs)
        # TODO: make it return a list so its more memory efficient (item_1.backward() and item_2.backward(), etc)
        d_loss = self.model.compute_discriminator_loss(imgs, fake_imgs)
        self.backward(d_loss, self.lr_scheduler_d, self.optimizer_d, should_step=is_last_accum_batch)

        time_end = time.time()

        lr_g = self.optimizer_g.param_groups[0]['lr']
        lr_d = self.optimizer_d.param_groups[0]['lr']
        it_s = imgs.size(0) / (time_end - time_start)
        metrics = {
            'g_loss': g_loss.item(), 'd_loss': d_loss.item(), 'lr_g': lr_g, 'lr_d': lr_d, 'its': it_s
        }
        return metrics

    def backward(self, loss: torch.Tensor, lr_scheduler: Scheduler, optimizer: torch.optim.Optimizer,
                 should_step: bool):
        accum_steps = self.configs.grad_accum_steps
        if accum_steps > 1:
            loss /= accum_steps

        loss.backward()

        if should_step:
            optimizer.step()
            optimizer.zero_grad()

            lr_scheduler.step(current_loss=loss.item())

    def evaluate(self, data: Optional[Dataset] = None) -> dict:
        data = self.load_dataset(self.configs, data, 'evaluation')

        self.before_evaluate(data)
        dataloader = data.train_dataloader()
        self.model.eval()

        if self.device.type.startswith('cuda') and not self.device.type.endswith(':0'):
            torch.cuda.set_device(self.device)

        metrics_value = compute_metrics(model=self.model, dataloader=dataloader, metrics=self.eval_metrics, config={
            'output_path': self.configs.output_path,
            'experiment': self.configs.project,
            'samples': 50_000,
            'batch_size': self.configs.batch_size,
            'device': self.device,
            'verbose': True,
        })

        return metrics_value
