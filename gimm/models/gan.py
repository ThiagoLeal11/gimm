""" Generative Adversarial Networks model implementation

GAN that utilizes the vit architecture for the generator and the discriminator.

GAN Implementation adapted from:
    https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/basic-gan.html

Papers:
    Generative Adversarial Networks - https://arxiv.org/abs/1406.2661
"""
from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor

from gimm.models.definition import ModuleGAN, ImageTensor, Loss, Logits, Size


class Discriminator(nn.Module):
    def __init__(self, in_features: Size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(in_features)), 512, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = torch.flatten(img, 1)
        validity = self.model(img_flat)

        return validity


class Generator(nn.Module):
    def __init__(self, latent_dim: int, in_features: Size):
        super().__init__()
        self.in_features = in_features

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128, bias=True),
            nn.LeakyReLU(0.2, True),

            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),

            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),

            nn.Linear(512, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, True),

            # Output: channels * image size * image size.
            nn.Linear(1024, int(np.prod(in_features)), bias=True),
            nn.Tanh(),
        )

    def forward(self, z: Tensor):
        img = self.model(z)
        img = img.view(img.size(0), *self.in_features)
        return img


class GAN(ModuleGAN):
    def __init__(self, in_features: Optional[Size] = None, latent_dim: int = 100):
        super().__init__()

        self.in_features = in_features
        self.latent_dim = latent_dim

    def construct(self, in_features: Size) -> 'GAN':
        in_features = in_features or self.in_features
        assert in_features is not None, "in_features must be provided"

        self.generator = Generator(self.latent_dim, in_features)
        self.discriminator = Discriminator(in_features)
        return self

    def forward(self, z):
        return self.generator(z)

    def get_latent(self, batch_size: int) -> Tensor:
        return torch.randn(batch_size, self.latent_dim)

    def generate_random_images(self, x: Tensor) -> ImageTensor:
        z = self.get_latent(x.shape[0]).type_as(x)
        return self.forward(z)

    def compute_loss(self, imgs: Tensor, labels: Tensor) -> tuple[Loss, Logits]:
        logits = self.discriminator(imgs)
        return bce(logits, labels), logits

    def compute_generator_loss(self, imgs: Tensor) -> tuple[Tensor, Tensor]:
        fake_imgs = self.generate_random_images(imgs)
        g_loss, _ = self.loss_to_real(fake_imgs)
        return g_loss, fake_imgs.detach()

    def compute_discriminator_loss(self, imgs: Tensor, fake_imgs: Tensor) -> Tensor:
        real_loss, _ = self.loss_to_real(imgs)
        fake_loss, _ = self.loss_to_fake(fake_imgs.detach())
        return real_loss + fake_loss

def bce(y_hat: Tensor, y: Tensor) -> Tensor:
    return torch.nn.functional.binary_cross_entropy(y_hat, y)
