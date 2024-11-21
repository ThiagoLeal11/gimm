import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from gimm.datasets.cifar10 import DatasetCifar10
from gimm.datasets.definition import Dataset
from gimm.models.vitgan import VitGAN


# TODO: checkpoint
# TODO: wandb
# TODO: gradacumm
# TODO: lr scheduler
# TODO: Mind the (optimality) gap: A Gap-Aware Learning Rate Scheduler for Adversarial Nets


@dataclass
class Config:
    input_size: list[int]
    batch_size: int = 64
    num_workers = 4
    project='gimm'
    lr: float = 0.0002
    b1: float = 0.5
    b2: float = 0.999
    epochs: int = 100
    grad_accum_steps: int = 1
    output_path: str = "output"
    compile_model = False


class BaseTrainer(ABC):
    def __init__(self, configs: Config):
        self.configs = configs
        self.model = VitGAN()
        self.fixed_z = self.model.get_latent(16)

        self.optimizer_g, self.optimizer_d = self.optimizers()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def optimizers(self):
        optimizer_g = torch.optim.Adam(self.model.generator.parameters(), lr=self.configs.lr, betas=(self.configs.b1, self.configs.b2))
        optimizer_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.configs.lr, betas=(self.configs.b1, self.configs.b2))
        return optimizer_g, optimizer_d

    def before_training(self):
        self.model.generator.train()
        self.model.discriminator.train()

        self.model.generator.to(self.device)
        self.model.discriminator.to(self.device)

    @abstractmethod
    def training_step(self, batch):
        pass

    def train(self, data: Dataset):
        self.before_training()

        train_dataloader = data.train_dataloader()
        for batch in train_dataloader:
            batch = [b.to(self.device) for b in batch]
            self.training_step(batch)


class Trainer(BaseTrainer):
    def training_step(self, batch):
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
        print({'g_loss': g_loss.item(), 'd_loss': d_loss.item(), 'lr': lr})
        print(f'it/s: {imgs.size(0) / (time_end - time_start)}')




def main():
    torch.set_float32_matmul_precision("medium")

    config = Config(
        # input_size=[1, 28, 28],
        input_size=[3, 32, 32],
    )

    dataset = DatasetCifar10(batch_size=config.batch_size, num_workers=config.num_workers)
    trainer = Trainer(config)
    trainer.train(dataset)


if __name__ == "__main__":
    main()
