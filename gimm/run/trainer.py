import time
from dataclasses import dataclass
from typing import Optional, Generator

import torch

from gimm.chekpoint import Checkpoint
from gimm.datasets.definition import Dataset
from gimm.eval.fidelity import FidelityEvalMetric, compute_metrics
from gimm.logs.log import Logger, get_wandb_run_id, LoggerWandb
from gimm.models.definition import ModuleGAN
from gimm.run.optimizer import Optimizer, Adam
from gimm.scheduler.constant import ConstantLR
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
    g_optimizer: Optimizer = Adam()
    d_optimizer: Optimizer = Adam()

    # Schedulers
    g_scheduler: Scheduler = ConstantLR()
    d_scheduler: Scheduler = ConstantLR()

    # Checkpoint
    output_path: str = "output"
    resume_checkpoint: str = ''  # 'output/checkpoints/checkpoint-150_000.pth.tar'

    # TODO: implement dataset in config option
    # Useful for distribute default configs for the model training
    dataset: Optional[str] = None

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
    def __init__(self, model: ModuleGAN, configs: TrainerConfig, loggers: list[Logger], eval_metrics: list[FidelityEvalMetric], device: torch.device = None):
        self.configs = configs
        self.loggers = loggers
        self.step = 0

        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fixed_z = self.model.get_latent(configs.images_to_log)

        # Define the optimizers and schedulers
        self.optimizer_g = configs.g_optimizer.construct(self.model.generator.parameters())
        self.optimizer_d = configs.d_optimizer.construct(self.model.discriminator.parameters())
        self.lr_scheduler_g = configs.g_scheduler.construct(optimizer=self.optimizer_g)
        self.lr_scheduler_d = configs.d_scheduler.construct(optimizer=self.optimizer_d)

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

    def save_checkpoint(self):
        self.checkpoint.save(step=self.step)

    def resume_checkpoint(self, path: str):
        self.step = self.checkpoint.load(path, should_resume_config=True)

    def before_training(self):
        self.model.generator.to(self.device)
        self.model.discriminator.to(self.device)

    def before_evaluate(self):
        self.model.generator.to(self.device)
        self.model.discriminator.to(self.device)

    def after_evaluate(self):
        pass

    def after_training(self):
        self.save_checkpoint()

    def get_image_examples(self):
        return self.model.generate_images(self.fixed_z.to(self.device))

    @staticmethod
    def infinite_dataloader(dataloader) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        while True:
            for (imgs, labels) in dataloader:
                if imgs.size(0) == dataloader.batch_size:
                    yield imgs, labels
                # Discard the last batch if it is not a full batch

    def train(self, data: Dataset):
        self.before_training()
        self.model.set_train()

        self.optimizer_g.zero_grad()
        self.optimizer_d.zero_grad()

        step = self.step
        train_dataloader = data.train_dataloader()

        for (imgs, labels) in self.infinite_dataloader(train_dataloader):
            metrics = {}
            for accum_idx in range(self.configs.grad_accum_steps):
                start_idx = accum_idx * self.configs.batch_size
                end_idx = start_idx + self.configs.batch_size

                # Execute the training step
                batch = (imgs[start_idx:end_idx].to(self.device), labels[start_idx:end_idx].to(self.device))
                metrics = self.training_step(accum_idx, batch)

            if step % self.configs.steps_log == 0:
                for logger in self.loggers:
                    train_metrics = {f'train_{k}': v for k, v in metrics.items()}
                    logger.log(step=step, metrics=train_metrics)

            if step % self.configs.steps_image == 0:
                generated_image_examples = self.get_image_examples()
                for logger in self.loggers:
                    logger.log_image(step=step, image=generated_image_examples, prefix='train')

            if step >= self.configs.steps_total:
                break

            if step % self.configs.steps_checkpoint == 0 and step > 0:
                print('Salvando checkpoint', self.step)
                self.save_checkpoint()

            if step % self.configs.steps_eval == 0 and step > 0:
                self.model.set_eval()
                metrics = self.evaluate(data)
                self.model.set_train()

                for logger in self.loggers:
                    eval_metrics = {f'eval_{k}': v for k, v in metrics.items()}
                    logger.log(step=step, metrics=eval_metrics)

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
        g_loss, fake_imgs = self.model.generator_loss(imgs)
        self.backward(g_loss, self.lr_scheduler_g, self.optimizer_g, should_step=is_last_accum_batch)

        # train discriminator
        # if accum_idx < accum_steps * self.configs.d_updates_per_step:
        #     if not fake_imgs:
        #         fake_imgs = self.model.generate_images(imgs)
        d_loss = self.model.discriminator_loss(imgs, fake_imgs)
        self.backward(d_loss, self.lr_scheduler_d, self.optimizer_d, should_step=is_last_accum_batch)

        time_end = time.time()

        lr_g = self.optimizer_g.param_groups[0]['lr']
        lr_d = self.optimizer_d.param_groups[0]['lr']
        it_s = imgs.size(0) / (time_end - time_start)
        metrics = {
            'g_loss': g_loss.item(), 'd_loss': d_loss.item(), 'lr_g': lr_g, 'lr_d': lr_d, 'it/s': it_s
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

    def evaluate(self, data: Dataset) -> dict:
        self.before_evaluate()
        dataloader = data.train_dataloader()
        self.model.set_eval()

        metrics_value = compute_metrics(model=self.model, dataloader=dataloader, metrics=self.eval_metrics, config={
            'output_path': self.configs.output_path,
            'experiment': self.configs.project,
            'samples': 50_000,
            'batch_size': self.configs.batch_size,
            'device': self.device,
            'verbose': True,
        })

        return metrics_value
