""" Copied from https://github.com/pytorch/examples/blob/main/dcgan/main.py """
from typing import Sequence

import torch
import torch.nn as nn

from torch import Tensor
from typing_extensions import Optional

from gimm.models.definition import ModuleGAN, SampleTensor, Loss, Logits, Size


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=128, channels=3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.latent_dim = latent_dim
        nc = channels

        # Dense (4 × 4 × 256) → Reshape to (4, 4, 256)
        self.dense = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 256),
            nn.ReLU(True),
            nn.BatchNorm1d(4 * 4 * 256),
        )

        self.conv = nn.Sequential(
            # state size: (256) x 4 x 4
            # Up Convolution (128 filters, 5 × 5 kernel, stride 2, ReLU, batchnorm)
            nn.ConvTranspose2d(256, 128, 5, 2, 2, output_padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            # state size: (128) x 8 x 8
            # Up Convolution (64 filters, 4 × 4 kernel, stride 2, ReLU, batchnorm)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            # state size: (64) x 16 x 16
            # Up Convolution (3 filters, 4 × 4 kernel, stride 2, Tanh)
            nn.ConvTranspose2d(64, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size: (nc) x 32 x 32
        )
        self.dense.apply(weights_init)
        self.conv.apply(weights_init)

    def forward(self, x):
        # x shape: (batch, latent_dim) or (batch, latent_dim, 1, 1)
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        x = self.dense(x)
        x = x.view(x.size(0), 256, 4, 4)
        return self.conv(x)


class DCGANDiscriminator(nn.Module):
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
            # Flatten → Dense (1 unit)
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid(),
        )
        self.main.apply(weights_init)

    def forward(self, x):
        output = self.main(x)
        return output.squeeze(1)


class DCGAN(ModuleGAN):
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
        self.discriminator = DCGANDiscriminator(channels=channels)

    def get_latent(self, batch_size: int) -> Tensor:
        return torch.randn(batch_size, self.latent_dim)

    def compute_loss(self, imgs: Tensor, labels: Tensor) -> tuple[Loss, Logits]:
        logits = self.discriminator(imgs)
        return torch.nn.functional.binary_cross_entropy(logits, labels), logits

    def compute_generator_loss(self, imgs: Tensor) -> Sequence[Tensor] | Tensor:
        fake_imgs = self.generate_random_samples(imgs)
        g_loss, _ = self.loss_to_real(fake_imgs)
        return g_loss

    def compute_discriminator_loss(self, imgs: Tensor, fake_imgs: Tensor) -> Sequence[Tensor] | Tensor:
        real_loss, _ = self.loss_to_real(imgs)
        fake_loss, _ = self.loss_to_fake(fake_imgs)
        return real_loss, fake_loss
