""" Copied from https://github.com/torchgan/torchgan/blob/master/torchgan/models/dcgan.py """

from math import ceil, log2

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DCGANGenerator", "DCGANDiscriminator"]

from torch import Tensor
from typing_extensions import Optional

from gimm.models.definition import ModuleGAN, ImageTensor, Loss, Logits, Size


class DCGANGenerator(nn.Module):
    r"""Deep Convolutional GAN (DCGAN) generator from
    `"Unsupervised Representation Learning With Deep Convolutional Generative Aversarial Networks
    by Radford et. al. " <https://arxiv.org/abs/1511.06434>`_ paper

    Args:
        encoding_dims (int, optional): Dimension of the encoding vector sampled from the noise prior.
        out_size (int, optional): Height and width of the input image to be generated. Must be at
            least 16 and should be an exact power of 2.
        out_channels (int, optional): Number of channels in the output Tensor.
        step_channels (int, optional): Number of channels in multiples of which the DCGAN steps up
            the convolutional features. The step up is done as dim :math:`z \rightarrow d \rightarrow
            2 \times d \rightarrow 4 \times d \rightarrow 8 \times d` where :math:`d` = step_channels.
        batchnorm (bool, optional): If True, use batch normalization in the convolutional layers of
            the generator.
        nonlinearity (torch.nn.Module, optional): Nonlinearity to be used in the intermediate
            convolutional layers. Defaults to ``LeakyReLU(0.2)`` when None is passed.
        last_nonlinearity (torch.nn.Module, optional): Nonlinearity to be used in the final
            convolutional layer. Defaults to ``Tanh()`` when None is passed.
        label_type (str, optional): The type of labels expected by the Generator. The available
            choices are 'none' if no label is needed, 'required' if the original labels are
            needed and 'generated' if labels are to be sampled from a distribution.
    """

    def __init__(self, encoding_dims=100, out_size=32, out_channels=3, step_channels=64, batchnorm=True,
                 nonlinearity=None, last_nonlinearity=None, label_type="none", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoding_dims = encoding_dims
        self.label_type = label_type

        if out_size < 16 or ceil(log2(out_size)) != log2(out_size):
            raise Exception(
                "Target Image Size must be at least 16*16 and an exact power of 2"
            )
        num_repeats = out_size.bit_length() - 4
        self.ch = out_channels
        self.n = step_channels
        self.n = step_channels
        use_bias = not batchnorm
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        last_nl = nn.Tanh() if last_nonlinearity is None else last_nonlinearity
        model = []
        d = int(self.n * (2 ** num_repeats))
        if batchnorm is True:
            model.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.encoding_dims, d, 4, 1, 0, bias=use_bias
                    ),
                    nn.BatchNorm2d(d),
                    nl,
                )
            )
            for i in range(num_repeats):
                model.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(d, d // 2, 4, 2, 1, bias=use_bias),
                        nn.BatchNorm2d(d // 2),
                        nl,
                    )
                )
                d = d // 2
        else:
            model.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.encoding_dims, d, 4, 1, 0, bias=use_bias
                    ),
                    nl,
                )
            )
            for i in range(num_repeats):
                model.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(d, d // 2, 4, 2, 1, bias=use_bias),
                        nl,
                    )
                )
                d = d // 2

        model.append(
            nn.Sequential(
                nn.ConvTranspose2d(d, self.ch, 4, 2, 1, bias=True), last_nl
            )
        )
        self.model = nn.Sequential(*model)
        self._weight_initializer()

    def _weight_initializer(self):
        r"""Default weight initializer for all generator models.
        Models that require custom weight initialization can override this method
        """
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, feature_matching=False):
        r"""Calculates the output tensor on passing the encoding ``x`` through the Generator.

        Args:
            x (torch.Tensor): A 2D torch tensor of the encoding sampled from a probability
                distribution.
            feature_matching (bool, optional): Returns the activation from a predefined intermediate
                layer.

        Returns:
            A 4D torch.Tensor of the generated image.
        """
        x = x.view(-1, x.size(1), 1, 1)
        return self.model(x)


class DCGANDiscriminator(nn.Module):
    r"""Deep Convolutional GAN (DCGAN) discriminator from
    `"Unsupervised Representation Learning With Deep Convolutional Generative Aversarial Networks
    by Radford et. al. " <https://arxiv.org/abs/1511.06434>`_ paper

    Args:
        in_size (int, optional): Height and width of the input image to be evaluated. Must be at
            least 16 and should be an exact power of 2.
        in_channels (int, optional): Number of channels in the input Tensor.
        step_channels (int, optional): Number of channels in multiples of which the DCGAN steps up
            the convolutional features. The step up is done as dim :math:`z \rightarrow d \rightarrow
            2 \times d \rightarrow 4 \times d \rightarrow 8 \times d` where :math:`d` = step_channels.
        batchnorm (bool, optional): If True, use batch normalization in the convolutional layers of
            the generator.
        nonlinearity (torch.nn.Module, optional): Nonlinearity to be used in the intermediate
            convolutional layers. Defaults to ``LeakyReLU(0.2)`` when None is passed.
        last_nonlinearity (torch.nn.Module, optional): Nonlinearity to be used in the final
            convolutional layer. Defaults to ``Tanh()`` when None is passed.
        label_type (str, optional): The type of labels expected by the Generator. The available
            choices are 'none' if no label is needed, 'required' if the original labels are
            needed and 'generated' if labels are to be sampled from a distribution.
    """

    def __init__(self, in_size=32, in_channels=3, step_channels=64, batchnorm=True, nonlinearity=None,
                 last_nonlinearity=None, label_type="none", *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.input_dims = in_channels
        self.label_type = label_type

        if in_size < 16 or ceil(log2(in_size)) != log2(in_size):
            raise Exception(
                "Input Image Size must be at least 16*16 and an exact power of 2"
            )
        num_repeats = in_size.bit_length() - 4
        self.n = step_channels
        use_bias = not batchnorm
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        last_nl = (
            nn.LeakyReLU(0.2)
            if last_nonlinearity is None
            else last_nonlinearity
        )
        d = self.n
        model = [
            nn.Sequential(
                nn.Conv2d(self.input_dims, d, 4, 2, 1, bias=True), nl
            )
        ]
        if batchnorm is True:
            for i in range(num_repeats):
                model.append(
                    nn.Sequential(
                        nn.Conv2d(d, d * 2, 4, 2, 1, bias=use_bias),
                        nn.BatchNorm2d(d * 2),
                        nl,
                    )
                )
                d *= 2
        else:
            for i in range(num_repeats):
                model.append(
                    nn.Sequential(
                        nn.Conv2d(d, d * 2, 4, 2, 1, bias=use_bias), nl
                    )
                )
                d *= 2
        self.disc = nn.Sequential(
            nn.Conv2d(d, 1, 4, 1, 0, bias=use_bias), last_nl
        )
        self.model = nn.Sequential(*model)
        self._weight_initializer()

    def _weight_initializer(self):
        r"""Default weight initializer for all generator models.
        Models that require custom weight initialization can override this method
        """
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, feature_matching=False):
        r"""Calculates the output tensor on passing the imTage ``x`` through the Discriminator.

        Args:
            x (torch.Tensor): A 4D torch tensor of the image.
            feature_matching (bool, optional): Returns the activation from a predefined intermediate
                layer.

        Returns:
            A 1D torch.Tensor of the probability of each image being real.
        """
        x = self.model(x)
        if feature_matching is True:
            return x
        else:
            x = self.disc(x)
            return x.view(x.size(0))


class DCGAN(ModuleGAN):
    def __init__(self, in_features: Optional[Size] = None, latent_dim: int = 100):
        super().__init__()

        self.in_features = in_features
        self.latent_dim = latent_dim

    def construct(self, in_features: Size) -> 'DCGAN':
        in_features = in_features or self.in_features
        assert in_features is not None, "in_features must be provided"

        assert len(in_features) == 3, "in_features must be a list of 3 integers (channels, height, width)"
        assert in_features[0] in [1, 3], "in_features[0] must be either 1 (grayscale) or 3 (RGB)"
        assert in_features[1] == in_features[2], "in_features[1] and in_features[2] must be equal (height and width)"
        channels, img_size = in_features[0], in_features[1]

        self.generator = DCGANGenerator(encoding_dims=self.latent_dim, out_size=img_size, out_channels=channels)
        self.discriminator = DCGANDiscriminator(in_size=img_size, in_channels=channels)
        return self

    def forward(self, z):
        return self.generator(z)

    def get_latent(self, batch_size: int) -> Tensor:
        return torch.randn(batch_size, self.latent_dim)

    def generate_images(self, x: Tensor) -> ImageTensor:
        z = self.get_latent(x.shape[0]).type_as(x)
        return self.forward(z)

    def loss(self, imgs: Tensor, labels: Tensor) -> tuple[Loss, Logits]:
        logits = self.discriminator(imgs).reshape(-1, 1)
        return bce(logits, labels), logits

    def generator_loss(self, imgs: Tensor) -> tuple[Tensor, Tensor]:
        fake_imgs = self.generate_images(imgs)
        g_loss, _ = self.real_loss(fake_imgs)
        return g_loss, fake_imgs.detach()

    def discriminator_loss(self, imgs: Tensor, fake_imgs: Tensor) -> Tensor:
        real_loss, _ = self.real_loss(imgs)
        fake_loss, _ = self.fake_loss(fake_imgs.detach())
        return real_loss + fake_loss

def bce(y_hat: Tensor, y: Tensor) -> Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)
