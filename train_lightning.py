import os
import sys
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
# from lightning.pytorch.callbacks import TQDMProgressBar
import torch
import torchvision
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
from pytorch_lightning.utilities.types import STEP_OUTPUT

from gimm.datasets.cifar10 import DatasetCifar10
from gimm.datasets.mnist import DatasetMNIST
from gimm.models.gan import GAN
from gimm.models.vitgan import VitGAN


class Config:
    data_dir: str = os.environ.get("PATH_DATASETS", ".")
    batch_size: int = 64
    num_workers = 4


class TrainModel(pl.LightningModule):
    def __init__(
        self,
        # model: ModuleGAN,
        channels: int,
        width: int,
        height: int,
    ):
        super().__init__()
        # self.model = model
        self.save_hyperparameters()
        self.automatic_optimization = False

        # networks
        data_shape = [channels, width, height]
        # self.model = GAN(in_features=data_shape, latent_dim=100)
        self.model = VitGAN()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        imgs, _ = batch

        optimizer_g, optimizer_d = self.optimizers()

        # train generator
        self.toggle_optimizer(optimizer_g)

        g_loss, fake_imgs = self.model.generator_loss(imgs)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()

        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        self.toggle_optimizer(optimizer_d)

        d_loss = self.model.discriminator_loss(imgs, fake_imgs)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()

        lr = optimizer_g.param_groups[0]['lr']
        self.log_dict({'g_loss': g_loss, 'd_loss': d_loss, 'lr': lr}, prog_bar=True)

        self.untoggle_optimizer(optimizer_d)

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch

        # with torch.no_grad:
        g_loss, fake_imgs = self.model.generator_loss(imgs)
        d_loss = self.model.discriminator_loss(imgs, fake_imgs)

        self.log_dict({"val_loss_G": g_loss, "val_loss_d": d_loss})
        self.log_images(fake_imgs, self.current_epoch)

    def configure_optimizers(self):
        # TODO: pegar das configs
        lr = 0.0002
        b1 = 0.5
        b2 = 0.99

        opt_g = torch.optim.Adam(self.model.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def log_images(self, imgs: torch.Tensor, epoch: int):
        sample_imgs = imgs[:16]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("train/generated_images", grid, epoch)

    # def on_validation_epoch_end(self):
    #     z = self.validation_z.type_as(self.model.generator.model[0].weight)
    #
    #     # log sampled images
    #     sample_imgs = self(z)
    #     grid = torchvision.utils.make_grid(sample_imgs)
    #     # self.logger.experiment.add_image("validation/generated_images", grid, self.current_epoch)


def main():
    config = Config()
    # dataset = DatasetMNIST(batch_size=config.batch_size, num_workers=config.num_workers)
    dataset = DatasetCifar10(batch_size=config.batch_size, num_workers=config.num_workers)

    # model = GAN(in_features=[1, 28, 28], latent_dim=100)
    model = TrainModel(
        channels=1,
        width=28,
        height=28,
        # latent_dim=100,
        # lr=0.0002,
        # b1=0.5,
        # b2=0.999,
        # batch_size=config.batch_size,
    )
    # Initialize a trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=100,
        callbacks=[CustomProgressBar()],
    )

    # Train the model ⚡
    trainer.fit(model, dataset)


class CustomProgressBar(TQDMProgressBar):
    BAR_FORMAT = "{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}],{rate_fmt} [{postfix}]"

    def init_train_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for training."""
        return Tqdm(
            ncols=250,
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            file=sys.stdout,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
        )

    def init_validation_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The train progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        return Tqdm(
            ncols=250,
            desc=self.validation_description,
            position=(2 * self.process_position + 1),
            disable=True,
            leave=True,
            file=sys.stdout,
            bar_format="{l_bar}{bar:50}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}],{rate_fmt} [{postfix}]",
        )

if __name__ == "__main__":
    main()
