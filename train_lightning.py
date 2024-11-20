from dataclasses import dataclass

import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.loggers import WandbLogger

from gimm.datasets.cifar10 import DatasetCifar10
from gimm.datasets.mnist import DatasetMNIST
from gimm.logs.progress import CustomProgressBar
from gimm.models.gan import GAN
from gimm.models.vitgan import VitGAN


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


class TrainModel(pl.LightningModule):
    def __init__(
        self,
        configs: Config,
    ):
        super().__init__()
        # self.model = model
        self.configs = configs

        self.save_hyperparameters(configs.__dict__)
        self.automatic_optimization = False

        self.model = GAN(in_features=configs.input_size, latent_dim=100)
        # self.model = VitGAN()

        self.fixed_z = self.model.get_latent(16)

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

        self.log_dict({"val_loss_g": g_loss, "val_loss_d": d_loss})

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.model.generator.parameters(), lr=self.configs.lr, betas=(self.configs.b1, self.configs.b2))
        opt_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=self.configs.lr, betas=(self.configs.b1, self.configs.b2))
        return [opt_g, opt_d], []

    def on_validation_epoch_end(self):
        device = list(self.model.generator.parameters())[0].device
        const_imgs = self.model.generate_images(self.fixed_z.to(device))
        sample_imgs = const_imgs[:16]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.log_image(key="generated_images", images=[grid], caption=[f"{self.current_epoch}"], step=self.current_epoch)
        # self.logger.experiment.add_image(f"{self.configs.output_path}/validation/generated_images", grid, self.current_epoch)


def main():
    config = Config(
        input_size=[1, 28, 28],
    )
    dataset = DatasetMNIST(batch_size=config.batch_size, num_workers=config.num_workers)
    # dataset = DatasetCifar10(batch_size=config.batch_size, num_workers=config.num_workers)

    logger = WandbLogger(project=config.project)

    # Initialize a trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=config.epochs,
        callbacks=[CustomProgressBar()],
        logger=logger,
        # precision="bf16-mixed",
    )

    # init the model directly on the device
    with trainer.init_module():
        # model = GAN(in_features=[1, 28, 28], latent_dim=100)
        model = TrainModel(config)

    # Train the model ⚡
    trainer.fit(model, dataset)


if __name__ == "__main__":
    main()
