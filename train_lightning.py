import os
from typing import Any

import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.utilities.types import STEP_OUTPUT

from datasets.cifar10 import DatasetCifar10
from gimm.models.gan import GAN
from models.vitgan import VitGAN


class Config:
    data_dir: str = os.environ.get("PATH_DATASETS", ".")
    batch_size: int = 50
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

        g_loss = self.model.generator_loss(imgs)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()

        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        self.toggle_optimizer(optimizer_d)

        d_loss = self.model.discriminator_loss(imgs)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()

        self.untoggle_optimizer(optimizer_d)

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        loss = self.model.loss(imgs, is_real=True)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        lr = 0.0002
        b1 = 0.
        b2 = 0.99

        opt_g = torch.optim.Adam(self.model.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

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
        channels=3,
        width=32,
        height=32,
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
        max_epochs=10,
    )

    # Train the model ⚡
    trainer.fit(model, dataset)


if __name__ == "__main__":
    main()
