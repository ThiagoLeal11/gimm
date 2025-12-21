""" Copied from https://github.com/pytorch/examples/blob/main/dcgan/main.py """
import pathlib
from dataclasses import dataclass
from typing import Sequence, Literal, Optional

import torch
import torch.nn as nn
import torchvision

from torch import Tensor

from gimm.models.definition import ModuleGAN, SampleTensor, Loss, Logits, Size

from gimm.models.gap_x.iia_new import compute_iia_heatmap


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
        self.features = nn.Sequential(
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
        )
        self.classifier = nn.Sequential(
            # Flatten → Dense (1 unit)
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 2),
        )
        self.features.apply(weights_init)
        self.classifier.apply(weights_init)

        # Hook for storing gradients for IIA (igual ao GradModel)
        self.gradients = None

    def forward(self, x):
        return self.classifier(self.features(x))

    def forward_classifier(self, x):
        return self.classifier(x)

    def forward_features(self, x):
        return self.features(x)


@dataclass
class ConfigIIA:
    # iia_loss_weight: w < 1: interpolates between GAN loss and IIA loss. w > 1: adds IIA loss to GAN loss.
    iia_loss_weight: float = 0.5
    iia_start_step: int = 20_000
    iia_warmup_steps: int = 10_000
    # Weights real and fake maps according to the probability of being real/fake
    iia_map_weighting: bool = True
    iia_maps: Literal['fake', 'real', 'both'] = 'both'
    iia_maps_save_interval: int = 20_000


class DCGAN(ModuleGAN):
    def __init__(
        self,
        in_features: Optional[Size] = None,
        latent_dim: int = 128,
        iia_config: Optional[ConfigIIA] = None,
        output_path: str = 'output'
    ):
        super().__init__()

        if not in_features:
            in_features = (3, 32, 32)

        self.in_features = in_features
        self.latent_dim = latent_dim
        self.iia_config = iia_config or ConfigIIA()
        self.output_path = output_path

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
        loss = torch.nn.functional.cross_entropy(logits, labels.long())
        return loss, logits

    def compute_generator_loss(self, imgs: Tensor) -> Sequence[Tensor] | Tensor:
        fake_imgs = self.generate_random_samples(imgs)
        g_loss, g_logits = self.loss_to_real(fake_imgs)

        # Log iia heatmaps
        if self.current_step % self.iia_config.iia_maps_save_interval == 0:
            self.iia_save_heatmaps(fake_imgs, imgs)

        # Skip IIA loss before start step
        if self.current_step < self.iia_config.iia_start_step:
            return g_loss

        # IIA Map Weighting
        real_prob = fake_prob = 1.0
        if self.iia_config.iia_map_weighting:
            real_prob = torch.softmax(g_logits, dim=1)[:, 0].detach()
            fake_prob = (1.0 - real_prob).detach()

        # IIA maps selection
        if self.iia_config.iia_maps == 'fake':
            fake_prob = 0.0
        elif self.iia_config.iia_maps == 'real':
            real_prob = 0.0

        # IIA loss
        fake_map, real_map = self.iia_compute_heatmap(fake_imgs)
        iia_loss = (
            (fake_map * fake_imgs).flatten(1).sum(1) * fake_prob
            - (real_map * fake_imgs).flatten(1).sum(1) * real_prob
        ).mean()

        weight_warmup = self._compute_iia_weight_warmup(self.current_step, self.iia_config)

        # Combine loss weights
        loss_weight = 1.0 - (self.iia_config.iia_loss_weight * weight_warmup)
        iia_weight = self.iia_config.iia_loss_weight * weight_warmup
        if self.iia_config.iia_loss_weight > 1.0:
            loss_weight = 1.0
            iia_weight = (self.iia_config.iia_loss_weight - 1.0) * weight_warmup

        # Join losses
        return g_loss * loss_weight, iia_loss * iia_weight


    def compute_discriminator_loss(self, imgs: Tensor, fake_imgs: Tensor) -> Sequence[Tensor] | Tensor:
        real_loss, _ = self.loss_to_real(imgs)
        fake_loss, _ = self.loss_to_fake(fake_imgs)
        return real_loss, fake_loss

    def iia_compute_heatmap(self, images: Tensor) -> tuple[Tensor, Tensor]:
        rel_fake = compute_iia_heatmap(self.discriminator, images.detach(), label=0)  # fake class
        rel_real = compute_iia_heatmap(self.discriminator, images.detach(), label=1)  # real class
        return rel_fake, rel_real

    def iia_save_heatmaps(self, fake_imgs: Tensor, real_imgs: Tensor):
        fake_map, real_map = self.iia_compute_heatmap(fake_imgs)
        real_as_fake_map = compute_iia_heatmap(self.discriminator, real_imgs.detach(), label=0)
        real_as_real_map = compute_iia_heatmap(self.discriminator, real_imgs.detach(), label=1)
        base_path = pathlib.Path(f'{self.output_path}/iia_heatmaps/')
        base_path.mkdir(parents=True, exist_ok=True)

        self._save_imgs_grid(base_path / f'{self.current_step}_fake_map.png', fake_map)
        self._save_imgs_grid(base_path / f'{self.current_step}_real_map.png', real_map)

        self._save_imgs_grid(base_path / f'{self.current_step}_real_as_fake_map.png', real_as_fake_map)
        self._save_imgs_grid(base_path / f'{self.current_step}_real_as_real_map.png', real_as_real_map)

        self._save_imgs_grid(base_path / f'{self.current_step}_fake_imgs.png', fake_imgs)
        self._save_imgs_grid(base_path / f'{self.current_step}_real_imgs.png', real_imgs)

    @staticmethod
    def _save_imgs_grid(path: pathlib.Path, imgs: Tensor):
        grid = torchvision.utils.make_grid(imgs, normalize=True, scale_each=True)
        torchvision.utils.save_image(grid, path)

    @staticmethod
    def _compute_iia_weight_warmup(step: int, config: ConfigIIA) -> float:
        if step < config.iia_start_step:
            return 0.0
        elif step >= config.iia_start_step + config.iia_warmup_steps:
            return 1.0
        else:
            return (step - config.iia_start_step) / config.iia_warmup_steps