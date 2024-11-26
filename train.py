import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from sympy.solvers.ode import infinitesimals
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from gimm.chekpoint import Checkpoint
from gimm.datasets.cifar10 import DatasetCifar10
from gimm.datasets.definition import Dataset
from gimm.datasets.mnist import DatasetMNIST
from gimm.logs.log import Logger, LoggerConsole, LoggerWandb, LoggerImageFile, get_wandb_run_id
from gimm.models.gan import GAN
from gimm.models.vitgan import VitGAN


# TODO: eval_every = 5_000

# TODO: gradacumm
# TODO: lr scheduler
# TODO: Mind the (optimality) gap: A Gap-Aware Learning Rate Scheduler for Adversarial Nets

# TODO: métricas de avaliação
    # TODO: FID
    # TODO: IS
    # TODO: CMMD


@dataclass
class Config:
    input_size: list[int]
    batch_size: int = 64
    num_workers = 4
    project='gimm'
    lr: float = 0.0002
    b1: float = 0.5
    b2: float = 0.999

    total_steps: int = 1_000_000
    save_every: int = 50_000
    log_every: int = 10_000
    eval_every: int = 50_000

    # TODO: implement a scalar for checkpointing?
    every_scalar: int = 1_000

    grad_accum_steps: int = 1
    output_path: str = "output"
    resume_checkpoint: str = ''  # 'output/checkpoints/checkpoint-150_000.pth.tar'
    compile_model = False

    def __post_init__(self):
        self.total_steps = self.total_steps // self.batch_size
        self.save_every = self.save_every // self.batch_size
        self.log_every = self.log_every // self.batch_size
        self.eval_every = self.eval_every // self.batch_size

        self.safe_log_every = self.log_every * self.batch_size


class BaseTrainer(ABC):
    def __init__(self, configs: Config, loggers: list[Logger]):
        self.configs = configs
        self.loggers = loggers
        self.step = 0


        # self.model = VitGAN()
        self.model = GAN(in_features=configs.input_size, latent_dim=100)


        self.fixed_z = self.model.get_latent(16)

        self.optimizer_g, self.optimizer_d = self.optimizers()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        # TODO: se o caminho já tiver modelos e não for resumir dar erro.
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

    @abstractmethod
    def training_step(self, batch) -> dict:
        pass

    # @abstractmethod
    # def evaluate_step(self, batch) -> dict:
    #     pass

    @staticmethod
    def infinite_dataloader(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    def train(self, data: Dataset):
        self.before_training()
        self.model.set_train()

        train_dataloader = data.train_dataloader()
        for batch in self.infinite_dataloader(train_dataloader):
            batch = [b.to(self.device) for b in batch]
            metrics = self.training_step(batch)

            for logger in self.loggers:
                train_metrics = {f'train_{k}': v for k, v in metrics.items()}
                logger.log(step=self.step, metrics=train_metrics)

            # TODO: isso pode causar problemas se o batch_size do resume não for o mesmo do checkpoint.
            batch_idx = self.step // self.configs.batch_size
            if batch_idx >= self.configs.total_steps:
                break

            if batch_idx % self.configs.save_every == 0:
                print('Salvando checkpoint')
                self.save_checkpoint()

            if batch_idx % self.configs.eval_every == 0:
                self.model.set_eval()
                self.evaluate()
                self.model.set_train()

            # All logs are taken from the start step.
            self.step += batch[0].size(0)

        self.after_training()

    # TODO: implement evaluate training loop
    def evaluate(self):
        pass


class Trainer(BaseTrainer):
    def training_step(self, batch) -> dict:
        imgs, labels = batch

        time_start = time.time()

        # train generator
        self.optimizer_g.zero_grad()
        g_loss, fake_imgs = self.model.generator_loss(imgs)
        g_loss.backward()
        self.optimizer_g.step()

        # train discriminator
        self.optimizer_d.zero_grad()
        d_loss = self.model.discriminator_loss(imgs, fake_imgs)
        d_loss.backward()
        self.optimizer_d.step()

        time_end = time.time()

        lr = self.optimizer_g.param_groups[0]['lr']
        metrics = {
            'g_loss': g_loss.item(), 'd_loss': d_loss.item(), 'lr': lr, 'it/s': imgs.size(0) / (time_end - time_start)
        }
        return metrics


def main():
    torch.set_float32_matmul_precision("medium")

    config = Config(
        input_size=[1, 28, 28],
        # input_size=[3, 32, 32],
    )

    # dataset = DatasetCifar10(batch_size=config.batch_size, num_workers=config.num_workers)
    dataset = DatasetMNIST(batch_size=config.batch_size, num_workers=config.num_workers)
    trainer = Trainer(
        config,
        loggers=[
            LoggerConsole(interval=config.safe_log_every),
            # LoggerImageFile(interval=config.safe_log_every, path=config.output_path, img_format='PNG'),
            LoggerWandb(experiment='test', interval=config.safe_log_every, config=config.__dict__)
        ]
    )

    trainer.train(dataset)


if __name__ == "__main__":
    main()
