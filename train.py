import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from sympy.solvers.ode import infinitesimals
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from gimm.datasets.cifar10 import DatasetCifar10
from gimm.datasets.definition import Dataset
from gimm.datasets.mnist import DatasetMNIST
from gimm.logs.log import Logger, LoggerConsole, LoggerWandb, LoggerImageFile
from gimm.models.gan import GAN
from gimm.models.vitgan import VitGAN


# TODO: save_every = 5_000
# TODO: eval_every = 5_000

# TODO: checkpoint
# TODO: gradacumm
# TODO: lr scheduler
# TODO: Mind the (optimality) gap: A Gap-Aware Learning Rate Scheduler for Adversarial Nets

# TODO: métricas de avaliação
    # TODO: FID
    # TODO: IS
    # TODO: CMMD
    # TODO:


@dataclass
class Config:
    input_size: list[int]
    batch_size: int = 64
    num_workers = 4
    project='gimm'
    lr: float = 0.0002
    b1: float = 0.5
    b2: float = 0.999

    total_steps: int = 100_000
    save_every: int = 5_000
    log_every: int = 1_000
    eval_every: int = 5_000

    grad_accum_steps: int = 1
    output_path: str = "output"
    compile_model = False


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

    # TODO: separar isso igual no torchgan
    def optimizers(self):
        optimizer_g = torch.optim.Adam(self.model.generator.parameters(), lr=self.configs.lr, betas=(self.configs.b1, self.configs.b2))
        optimizer_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.configs.lr, betas=(self.configs.b1, self.configs.b2))
        return optimizer_g, optimizer_d

    def before_training(self):
        self.model.generator.to(self.device)
        self.model.discriminator.to(self.device)

    def set_model_training(self):
        self.model.generator.train()
        self.model.discriminator.train()

    def set_model_eval(self):
        self.model.generator.eval()
        self.model.discriminator.eval()

    @abstractmethod
    def training_step(self, batch) -> dict:
        pass

    def after_training(self):
        self.save_checkpoint()

    def save_checkpoint(self):
        pass
        # torch.save(self.model.state_dict(), 'checkpoint.pth')

    def evaluate(self):
        pass


    @staticmethod
    def infinite_dataloader(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    def train(self, data: Dataset):
        self.before_training()
        self.set_model_training()

        train_dataloader = data.train_dataloader()
        for batch_idx, batch in enumerate(self.infinite_dataloader(train_dataloader)):
            batch = [b.to(self.device) for b in batch]
            metrics = self.training_step(batch)

            self.step += batch[0].size(0)
            for logger in self.loggers:
                train_metrics = {f'train_{k}': v for k, v in metrics.items()}
                logger.log(step=self.step, metrics=train_metrics)

            if batch_idx >= self.configs.total_steps:
                break

            if batch_idx % self.configs.save_every == 0:
                self.save_checkpoint()

            if batch_idx % self.configs.eval_every == 0:
                self.set_model_eval()
                self.evaluate()
                self.set_model_training()

        self.after_training()


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
            LoggerConsole(interval=config.log_every),
            # LoggerImageFile(interval=config.log_every, path=config.output_path, img_format='PNG'),
            # LoggerWandb(experiment='test', interval=config.log_every, config=config.__dict__)
        ]
    )

    trainer.train(dataset)


if __name__ == "__main__":
    main()
