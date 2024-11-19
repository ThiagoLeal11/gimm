""" Generative Adversarial Networks model implementation

GAN that utilizes the vit architecture for the generator and the discriminator.

GAN Implementation adapted from:
    https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/basic-gan.html

Papers:
    Generative Adversarial Networks - https://arxiv.org/abs/1406.2661
"""


import numpy as np
import torch
from torch import nn, Tensor

from gimm.models.definition import ModuleGAN

ShapeType = list[int]

class Discriminator(nn.Module):
    def __init__(self, in_features: ShapeType):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(in_features)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class Generator(nn.Module):
    def __init__(self, latent_dim: int, in_features: ShapeType):
        super().__init__()
        self.in_features = in_features

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(in_features))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.in_features)
        return img


class GAN(ModuleGAN):
    def __init__(self, in_features: ShapeType, latent_dim: int = 100):
        super().__init__()

        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, in_features)
        self.discriminator = Discriminator(in_features)

    def forward(self, z):
        return self.generator(z)

    def _generate_images(self, x: Tensor) -> Tensor:
        z = torch.randn(x.shape[0], self.latent_dim).type_as(x)
        return self.forward(z)

    def loss(self, x: Tensor, is_real: bool) -> Tensor:
        y = torch.zeros(x.size(0), 1).type_as(x)
        if is_real:
            y = torch.ones(x.size(0), 1).type_as(x)

        return bce(self.discriminator(x), y)

    def generator_loss(self, imgs: Tensor) -> tuple[Tensor, Tensor]:
        fake_imgs = self._generate_images(imgs)
        g_loss = self.loss(fake_imgs, is_real=True)
        return g_loss, fake_imgs.detach()

    def discriminator_loss(self, imgs: Tensor, fake_imgs: Tensor) -> Tensor:
        real_loss = self.loss(imgs, is_real=True)
        fake_loss = self.loss(fake_imgs.detach(), is_real=False)
        return (real_loss + fake_loss) / 2

def bce(y_hat, y):
    return torch.nn.functional.binary_cross_entropy(y_hat, y)
