"""
LSGAN (Least Squares GAN) implementation based on DCGAN architecture.

Reference: "Least Squares Generative Adversarial Networks" by Mao et al.
https://arxiv.org/abs/1611.04076

Loss functions:
    L_D = E[(D(x) - 1)^2] + E[D(G(z))^2]
    L_G = E[(D(G(z)) - 1)^2]
"""
from typing import Sequence

import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional

from gimm.models.definition import ModuleGAN, Loss, Logits, Size
from gimm.models.gap_aware.dcgan import DCGANGenerator, weights_init


class LSGANDiscriminator(nn.Module):
    def __init__(self, channels=3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        nc = channels
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            # Convolution (64 filters, 5 × 5 kernel, stride 2, leaky ReLU, batchnorm, dropout 0.3)
            nn.Conv2d(nc, 64, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),
            # state size: (64) x 16 x 16
            # Convolution (128 filters, 5 × 5 kernel, stride 2, Leaky ReLU, batchnorm, dropout 0.3)
            nn.Conv2d(64, 128, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3),
            # state size: (128) x 8 x 8
            # Convolution (256 filters, 5 × 5 kernel, stride 2, Leaky ReLU, batchnorm, dropout 0.3)
            nn.Conv2d(128, 256, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),
            # state size: (256) x 4 x 4
            # Flatten → Dense (1 unit) - No Sigmoid for LSGAN
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
        )
        self.main.apply(weights_init)

    def forward(self, x):
        output = self.main(x)
        return output.squeeze(1)


class LSGAN(ModuleGAN):
    """
    Least Squares GAN (LSGAN) implementation.

    LSGAN replaces the cross-entropy loss with least squares loss,
    which helps stabilize training and generate higher quality images.

    Loss functions:
        L_D = E[(D(x) - 1)^2] + E[D(G(z))^2]
        L_G = E[(D(G(z)) - 1)^2]
    """

    def __init__(self, in_features: Optional[Size] = None, latent_dim: int = 128):
        super().__init__()

        if not in_features:
            in_features = (3, 32, 32)

        self.in_features = in_features
        self.latent_dim = latent_dim

        assert len(in_features) == 3, "in_features must be a list of 3 integers (channels, height, width)"
        assert in_features[0] in [1, 3], "in_features[0] must be either 1 (grayscale) or 3 (RGB)"
        assert in_features[1] == in_features[2], "in_features[1] and in_features[2] must be equal (height and width)"
        channels, img_size = in_features[0], in_features[1]

        self.generator = DCGANGenerator(latent_dim=self.latent_dim, channels=channels)
        self.discriminator = LSGANDiscriminator(channels=channels)

    def get_latent(self, batch_size: int) -> Tensor:
        return torch.randn(batch_size, self.latent_dim)

    def compute_loss(self, imgs: Tensor, labels: Tensor) -> tuple[Loss, Logits]:
        """
        Compute MSE loss for LSGAN.
        labels=1 for real, labels=0 for fake.
        """
        logits = self.discriminator(imgs)
        # MSE loss: (D(x) - label)^2
        loss = torch.mean((logits - labels) ** 2)
        return loss, logits

    def compute_generator_loss(self, imgs: Tensor) -> Sequence[Tensor] | Tensor:
        """
        Generator loss: E[(D(G(z)) - 1)^2]
        The generator wants the discriminator to output 1 (real) for fake images.
        """
        fake_imgs = self.generate_random_samples(imgs)
        logits = self.discriminator(fake_imgs)
        # Generator wants D(fake) to be 1
        g_loss = torch.mean((logits - 1) ** 2)
        return g_loss

    def compute_discriminator_loss(self, imgs: Tensor, fake_imgs: Tensor) -> Sequence[Tensor] | Tensor:
        """
        Discriminator loss: E[(D(x) - 1)^2] + E[D(G(z))^2]
        The discriminator wants to output 1 for real and 0 for fake.
        """
        # Real loss: (D(real) - 1)^2
        real_logits = self.discriminator(imgs)
        real_loss = torch.mean((real_logits - 1) ** 2)

        # Fake loss: D(fake)^2
        fake_logits = self.discriminator(fake_imgs.detach())
        fake_loss = torch.mean(fake_logits ** 2)

        return real_loss, fake_loss
