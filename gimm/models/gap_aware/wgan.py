"""
WGAN (Wasserstein GAN) implementation based on DCGAN architecture.

Reference: "Wasserstein GAN" by Arjovsky et al.
https://arxiv.org/abs/1701.07875

Loss functions:
    L_D = -E[D(x)] + E[D(G(z))]  (Critic wants to maximize D(real) - D(fake))
    L_G = -E[D(G(z))]            (Generator wants to maximize D(fake))

Note: In WGAN, the discriminator is called "critic" and outputs unbounded values.
Weight clipping is typically used to enforce the Lipschitz constraint.
"""
from typing import Sequence, Optional

import torch
import torch.nn as nn

from torch import Tensor

from gimm.models.definition import ModuleGAN, Loss, Logits, Size
from gimm.models.gap_aware.dcgan import DCGANGenerator, weights_init


class WGANCritic(nn.Module):
    """
    WGAN Critic (Discriminator) based on DCGAN architecture (TF-GAN style).

    Key differences from DCGAN discriminator:
    1. No Sigmoid activation at the output (unbounded output)
    2. No BatchNorm/InstanceNorm (original WGAN paper doesn't use normalization in critic)

    Architecture:
        Convolution (64 filters, 5 × 5 kernel, stride 2, leaky ReLU) →
        Convolution (128 filters, 5 × 5 kernel, stride 2, Leaky ReLU) →
        Convolution (256 filters, 5 × 5 kernel, stride 2, Leaky ReLU) →
        Flatten → Dense (1 unit).
    """

    def __init__(self, channels=3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        nc = channels
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            # Convolution (64 filters, 5 × 5 kernel, stride 2, leaky ReLU)
            nn.Conv2d(nc, 64, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (64) x 16 x 16
            # Convolution (128 filters, 5 × 5 kernel, stride 2, Leaky ReLU)
            nn.Conv2d(64, 128, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (128) x 8 x 8
            # Convolution (256 filters, 5 × 5 kernel, stride 2, Leaky ReLU)
            nn.Conv2d(128, 256, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (256) x 4 x 4
            # Flatten → Dense (1 unit) - No Sigmoid for WGAN
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
        )
        self.main.apply(weights_init)

    def forward(self, x):
        output = self.main(x)
        return output.squeeze(1)


class WGAN(ModuleGAN):
    """
    Wasserstein GAN (WGAN) implementation.

    WGAN uses the Wasserstein distance (Earth Mover's distance) instead of
    JS divergence, which provides more stable training and meaningful loss values.

    Loss functions:
        L_D = -E[D(x)] + E[D(G(z))]
        L_G = -E[D(G(z))]

    Note: This implementation includes weight clipping to enforce the Lipschitz
    constraint. For better results, consider using WGAN-GP (gradient penalty).
    """

    def __init__(
        self,
        in_features: Optional[Size] = None,
        latent_dim: int = 128,
        clip_value: float = 0.01
    ):
        super().__init__()

        if not in_features:
            in_features = (3, 32, 32)

        self.in_features = in_features
        self.latent_dim = latent_dim
        self.clip_value = clip_value

        assert len(in_features) == 3, "in_features must be a list of 3 integers (channels, height, width)"
        assert in_features[0] in [1, 3], "in_features[0] must be either 1 (grayscale) or 3 (RGB)"
        assert in_features[1] == in_features[2], "in_features[1] and in_features[2] must be equal (height and width)"
        channels, img_size = in_features[0], in_features[1]

        self.generator = DCGANGenerator(latent_dim=self.latent_dim, channels=channels)
        self.discriminator = WGANCritic(channels=channels)

    def get_latent(self, batch_size: int) -> Tensor:
        return torch.randn(batch_size, self.latent_dim)

    def compute_loss(self, imgs: Tensor, labels: Tensor) -> tuple[Loss, Logits]:
        """
        Compute Wasserstein loss.
        Note: labels are not used in traditional sense for WGAN.
        labels=1 means we want high critic output (real)
        labels=0 means we want low critic output (fake)
        """
        logits = self.discriminator(imgs)
        # For WGAN: if label=1 (real), we want to maximize D(x), so loss = -D(x)
        # if label=0 (fake), we want to minimize D(x), so loss = D(x)
        # This can be written as: loss = (1 - 2*label) * D(x) = -label * D(x) + (1-label) * D(x)
        loss = torch.mean((1 - 2 * labels) * logits)
        return loss, logits

    def compute_generator_loss(self, imgs: Tensor) -> Sequence[Tensor] | Tensor:
        """
        Generator loss: -E[D(G(z))]
        The generator wants to maximize the critic's output for fake images.
        """
        fake_imgs = self.generate_random_samples(imgs)
        logits = self.discriminator(fake_imgs)
        # Generator wants to maximize D(fake), so minimize -D(fake)
        g_loss = -torch.mean(logits)
        return g_loss

    def compute_discriminator_loss(self, imgs: Tensor, fake_imgs: Tensor) -> Sequence[Tensor] | Tensor:
        """
        Critic loss: -E[D(x)] + E[D(G(z))]
        The critic wants to maximize D(real) - D(fake).
        We return the loss to minimize, so: -D(real) + D(fake)
        """
        # Real: want to maximize D(real), so minimize -D(real)
        real_logits = self.discriminator(imgs)
        real_loss = -torch.mean(real_logits)

        # Fake: want to minimize D(fake), so minimize D(fake)
        fake_logits = self.discriminator(fake_imgs)
        fake_loss = torch.mean(fake_logits)

        return real_loss, fake_loss

    def after_discriminator_step(self):
        # Clip critic weights to enforce Lipschitz constraint.
        for p in self.discriminator.parameters():
            p.data.clamp_(-self.clip_value, self.clip_value)
