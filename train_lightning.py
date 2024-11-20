from dataclasses import dataclass

import pytorch_lightning as pl
import torch
import torchvision

from gimm.datasets.cifar10 import DatasetCifar10
from gimm.logs.progress import CustomProgressBar
from gimm.models.vitgan import VitGAN


@dataclass
class Config:
    input_size: list[int]
    batch_size: int = 64
    num_workers = 4
    lr: float = 0.0002
    b1: float = 0.5
    b2: float = 0.999
    epochs: int = 100
    grad_accum_steps: int = 1


class TrainModel(pl.LightningModule):
    def __init__(
        self,
        configs: Config
    ):
        super().__init__()
        # self.model = model
        self.configs = configs
        # self.save_hyperparameters()
        self.automatic_optimization = False

        # self.model = GAN(in_features=configs.input_size, latent_dim=100)
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
        opt_g = torch.optim.Adam(self.model.generator.parameters(), lr=self.configs.lr, betas=(self.configs.b1, self.configs.b2))
        opt_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.configs.lr, betas=(self.configs.b1, self.configs.b2))
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
    config = Config(
        input_size=[3, 32, 32],
    )
    # dataset = DatasetMNIST(batch_size=config.batch_size, num_workers=config.num_workers)
    dataset = DatasetCifar10(batch_size=config.batch_size, num_workers=config.num_workers)

    # model = GAN(in_features=[1, 28, 28], latent_dim=100)
    model = TrainModel(config)
    # Initialize a trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=config.epochs,
        callbacks=[CustomProgressBar()],
    )

    # Train the model ⚡
    trainer.fit(model, dataset)


if __name__ == "__main__":
    main()
