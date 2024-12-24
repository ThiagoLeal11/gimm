import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator

import torch

from gimm.chekpoint import Checkpoint
from gimm.datasets.cifar10 import DatasetCifar10
from gimm.datasets.definition import Dataset
from gimm.datasets.mnist import DatasetMNIST
from gimm.eval.metrics.fid import compute_fid
from gimm.logs.log import Logger, LoggerConsole, LoggerWandb, LoggerImageFile, get_wandb_run_id
from gimm.models.definition import ModuleGAN
from gimm.models.gan import GAN
from gimm.models.vitgan import VitGAN
from gimm.scheduler.constant import ConstantLR
from gimm.scheduler.scheduler import Scheduler


# TODO: eval_every = 5_000

# TODO: lr scheduler
# TODO: Mind the (optimality) gap: A Gap-Aware Learning Rate Scheduler for Adversarial Nets

# TODO: métricas de avaliação
    # TODO: FID
    # TODO: IS
    # TODO: CMMD


# TODO: add continuous shufflling for dataset (seed += 1 for each rerun)

# Implement adam for classifier and adabelief for generator https://arxiv.org/pdf/2411.03999v1 (verify FID)
# Implement AdaBelief https://arxiv.org/pdf/2010.07468


@dataclass
class Config:
    input_size: list[int]
    batch_size: int = 128
    num_workers = 4
    project='gimm'
    lr: float = 0.0002
    b1: float = 0.5
    b2: float = 0.999

    # TODO: implement a scalar for checkpointing?
    every_scalar: int = 1_000

    grad_accum_steps: int = 1
    output_path: str = "output"
    resume_checkpoint: str = ''  # 'output/checkpoints/checkpoint-150_000.pth.tar'
    compile_model = False

    log_train_images: int = 128

    # Training controls
    epochs: int = 128
    checkpoint_freq : float = 1  # 1.0 means save every epoch
    log_freq: float = 0.1  # 0.1 means log every 10% of the total steps
    log_image_freq: float = 1  # 1.0 means log every epoch

    # Private vars
    steps_total: int = None
    steps_save: int = None
    steps_log: int = None
    steps_image: int = None
    steps_eval: int = None

    def __post_init__(self):
        self.full_batch_size = self.batch_size * self.grad_accum_steps

        # TODO: better name
        self.max_updates_adversaries = max(self.g_updates_per_step, self.d_updates_per_step)

    def adjust_to_batch(self, value: float) -> int:
        return math.floor(value // self.full_batch_size) * self.full_batch_size

    def compute_controls(self, dataset_size: int):
        # Always drop the last batch if it is not a full batch_size
        steps_per_epoch = self.adjust_to_batch(dataset_size)

        self.steps_total = self.epochs * steps_per_epoch
        self.steps_save = self.adjust_to_batch(steps_per_epoch * self.checkpoint_freq)
        self.steps_log = self.adjust_to_batch(steps_per_epoch * self.log_freq)
        self.steps_image = self.adjust_to_batch(steps_per_epoch * self.log_image_freq)
        self.steps_eval = self.adjust_to_batch(steps_per_epoch * self.log_image_freq)


class BaseTrainer(ABC):
    def __init__(self, model: ModuleGAN, configs: Config, loggers: list[Logger], device: torch.device = None):
        self.configs = configs
        self.loggers = loggers
        self.step = 0

        self.model = model
        self.fixed_z = self.model.get_latent(configs.log_train_images)

        self.optimizer_g, self.optimizer_d = self.optimizers()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.lr_scheduler_g = ConstantLR(optimizer=self.optimizer_g)
        self.lr_scheduler_d = ConstantLR(optimizer=self.optimizer_d)

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

    # TODO: separar isso igual no torchgan
    def optimizers(self):
        optimizer_g = torch.optim.Adam(self.model.generator.parameters(), lr=self.configs.lr, betas=(self.configs.b1, self.configs.b2))
        optimizer_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.configs.lr, betas=(self.configs.b1, self.configs.b2))
        return optimizer_g, optimizer_d

    def save_checkpoint(self):
        self.checkpoint.save(step=self.step)

    def resume_checkpoint(self, path: str):
        self.step = self.checkpoint.load(path, should_resume_config=True)

    def before_training(self):
        self.model.generator.to(self.device)
        self.model.discriminator.to(self.device)

    def before_evaluate(self):
        pass

    def after_evaluate(self):
        pass

    def after_training(self):
        self.save_checkpoint()

    def get_image_examples(self):
        return self.model.generate_images(self.fixed_z.to(self.device))

    @abstractmethod
    def training_step(self, accum_idx: int, batch: tuple[torch.Tensor, torch.Tensor]) -> dict:
        pass

    # @abstractmethod
    # def evaluate_step(self, batch) -> dict:
    #     pass

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

        # Adjust the steps to the dataset
        self.configs.compute_controls(len(train_dataloader.dataset))

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

            if step % self.configs.steps_save == 0 and step > 0:
                print('Salvando checkpoint', self.step)
                self.save_checkpoint()

            if step % self.configs.steps_eval == 0:
                self.model.set_eval()
                self.evaluate(data)
                self.model.set_train()

            # All logs are taken from the start step.
            self.step += imgs.size(0)
            step = self.step

        self.after_training()

    # TODO: implement evaluate training loop
    def evaluate(self, data: Dataset):
        self.before_evaluate()
        dataloader = data.train_dataloader()
        self.model.set_eval()

        # get 50.000 real images
        real_imgs = None
        for imgs, _ in dataloader:
            if real_imgs is None:
                real_imgs = imgs

            remaining = 1_000 - real_imgs.size(0)
            if remaining <= 0:
                break

            real_imgs = torch.cat([real_imgs, imgs[:remaining]], dim=0)

        # get 50.000 fake images
        latent = self.model.get_latent(1_000)
        fake_imgs = self.model.generate_images(latent)

        real_imgs = real_imgs.view(1_000, 1, 28, 28).expand(-1, 3, -1, -1)
        fake_imgs = fake_imgs.view(1_000, 1, 28, 28).expand(-1, 3, -1, -1)

        fid = compute_fid(real_imgs, fake_imgs)
        print(f'FID: {fid}')




# TODO: implement d_updates_per_step and g_updates_per_step to control the number of updates per step per adversarial.
class Trainer(BaseTrainer):
    def training_step(self, accum_idx: int, batch: tuple[torch.Tensor, torch.Tensor]) -> dict:
        is_last_accum_batch = (accum_idx + 1) % self.configs.grad_accum_steps == 0
        imgs, labels = batch

        time_start = time.time()

        # train generator
        g_loss, fake_imgs = self.model.generator_loss(imgs)
        self.backward(g_loss, self.lr_scheduler_g, self.optimizer_g, should_step=is_last_accum_batch)

        # train discriminator
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

    # TODO: create a custom optimizer to join optimizers and lr_schedulers
    def backward(self, loss: torch.Tensor, lr_scheduler: Scheduler, optimizer: torch.optim.Optimizer, should_step: bool):
        accum_steps = self.configs.grad_accum_steps
        if accum_steps > 1:
            loss /= accum_steps

        loss.backward()

        if should_step:
            optimizer.step()
            optimizer.zero_grad()

            lr_scheduler.step(current_loss=loss.item())


def main():
    torch.set_float32_matmul_precision("medium")

    config = Config(
        input_size=[1, 28, 28],
        # input_size=[3, 32, 32],
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset = DatasetCifar10(batch_size=config.batch_size, num_workers=config.num_workers)
    dataset = DatasetMNIST(batch_size=config.full_batch_size, num_workers=config.num_workers, pin_memory_device=device)

    trainer = Trainer(
        # model = VitGAN()
        model = GAN(in_features=config.input_size),
        configs=config,
        loggers=[
            LoggerConsole(),
            LoggerImageFile(path=config.output_path, img_format='PNG'),
            # LoggerWandb(experiment='test', interval=config.safe_log_every, config=config.__dict__)
        ],)

    trainer.train(dataset)


if __name__ == "__main__":
    main()
