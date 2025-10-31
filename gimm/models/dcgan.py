""" Copied from https://github.com/pytorch/examples/blob/main/dcgan/main.py """

from math import ceil, log2

import torch
import torch.nn as nn

from torch import Tensor
from typing_extensions import Optional

from gimm.models.definition import ModuleGAN, ImageTensor, Loss, Logits, Size


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class DCGANGenerator(nn.Module):
    r"""Deep Convolutional GAN (DCGAN) generator from
    `"Unsupervised Representation Learning With Deep Convolutional Generative Aversarial Networks
    by Radford et. al. " <https://arxiv.org/abs/1511.06434>`_ paper

    Args:
        latent_dim (int, optional): Dimension of the encoding vector sampled from the noise prior.
        channels (int, optional): Number of channels in the output Tensor.
        filters (int, optional): Number of channels in multiples of which the DCGAN steps up
            the convolutional features. The step up is done as dim :math:`z \rightarrow d \rightarrow
            2 \times d \rightarrow 4 \times d \rightarrow 8 \times d` where :math:`d` = step_channels.
    """

    def __init__(self, latent_dim=100, channels=3, filters=64, *args, **kwargs):
        super().__init__(*args, **kwargs)

        nz = latent_dim
        ngf = filters
        nc = channels
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
        )
        self.main.apply(weights_init)

    def forward(self, x):
        return self.main(x)


class DCGANDiscriminator(nn.Module):
    r"""Deep Convolutional GAN (DCGAN) discriminator from
    `"Unsupervised Representation Learning With Deep Convolutional Generative Aversarial Networks
    by Radford et. al. " <https://arxiv.org/abs/1511.06434>`_ paper

    Args:
        channels (int, optional): Number of channels in the input Tensor.
        filters (int, optional): Number of channels in multiples of which the DCGAN steps up
            the convolutional features. The step up is done as dim :math:`z \rightarrow d \rightarrow
            2 \times d \rightarrow 4 \times d \rightarrow 8 \times d` where :math:`d` = step_channels.
    """

    def __init__(self, channels=3, filters=64, *args, **kwargs):
        super().__init__(*args, **kwargs)

        nc = channels
        ndf = filters
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )
        self.main.apply(weights_init)

    def forward(self, x):
        output = self.main(x)
        return output.view(-1, 1).squeeze(1)


class DCGAN(ModuleGAN):
    def __init__(self, in_features: Optional[Size] = None, latent_dim: int = 100):
        super().__init__()

        if not in_features:
            in_features = (3, 64, 64)

        self.in_features = in_features
        self.latent_dim = latent_dim

        assert len(in_features) == 3, "in_features must be a list of 3 integers (channels, height, width)"
        assert in_features[0] in [1, 3], "in_features[0] must be either 1 (grayscale) or 3 (RGB)"
        assert in_features[1] == in_features[2], "in_features[1] and in_features[2] must be equal (height and width)"
        channels, img_size = in_features[0], in_features[1]

        self.generator = DCGANGenerator(latent_dim=self.latent_dim, channels=channels)
        self.discriminator = DCGANDiscriminator(channels=channels)

    def forward(self, z):
        return self.generator(z)

    def get_latent(self, batch_size: int) -> Tensor:
        return torch.randn(batch_size, self.latent_dim, 1, 1)

    def generate_random_images(self, x: Tensor) -> ImageTensor:
        z = self.get_latent(x.shape[0]).type_as(x)
        return self.forward(z)

    def compute_loss(self, imgs: Tensor, labels: Tensor) -> tuple[Loss, Logits]:
        logits = self.discriminator(imgs)
        return torch.nn.functional.binary_cross_entropy(logits, labels), logits

    def compute_generator_loss(self, imgs: Tensor) -> tuple[Tensor, Tensor]:
        fake_imgs = self.generate_random_images(imgs)
        g_loss, _ = self.loss_to_real(fake_imgs)
        return g_loss, fake_imgs.detach()

    def compute_discriminator_loss(self, imgs: Tensor, fake_imgs: Tensor) -> Tensor:
        real_loss, _ = self.loss_to_real(imgs)
        fake_loss, _ = self.loss_to_fake(fake_imgs.detach())
        return real_loss + fake_loss
