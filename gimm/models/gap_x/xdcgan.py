""" Copied from https://github.com/pytorch/examples/blob/main/dcgan/main.py """
import pathlib
from typing import Sequence

import torch
import torch.nn as nn
import torchvision

from torch import Tensor
from typing_extensions import Optional

from gimm.models.definition import ModuleGAN, SampleTensor, Loss, Logits, Size
from gimm.models.xdcgan import compute_iia_heatmap
import torch.nn.functional as F


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

    def forward(self, x, hook=False):
        if len(list(x.shape)) > 4:
            x = x.squeeze(1)

        x = self.features(x)
        if hook:
            x.register_hook(self.activations_hook)

        output = self.classifier(x)
        return output.view(-1, 2)  # [batch_size, 2]

    def forward_from_activations(self, x, hook=True):
        """
        Forward pass a partir das ativações (para IIA).
        Equivalente a forward com only_post_features=True no GradModel.

        Args:
            x: Tensor de ativações
            hook: Se True, registra hook para capturar gradientes

        Nota: No GradModel original, há post_features + relu + avgpool + flatten antes do classifier.
        No DCGAN, as features já terminam com LeakyReLU, então não aplicamos ReLU adicional.
        """
        if len(list(x.shape)) > 4:
            x = x.squeeze(1)

        if hook:
            x.register_hook(self.activations_hook)

        output = self.classifier(x)
        output = output.view(-1, 2)  # [batch_size, 2]
        return output

    def activations_hook(self, grad):
        """Hook para capturar gradientes das ativações."""
        self.gradients = grad.detach().clone()

    def get_activations_gradient(self):
        """Retorna os gradientes das ativações."""
        return self.gradients

    def get_activations(self, x):
        """Extrai as ativações da camada de features."""
        return self.features(x)


class DCGAN(ModuleGAN):
    def __init__(self, in_features: Optional[Size] = None, latent_dim: int = 128):
        super().__init__()

        if not in_features:
            in_features = (3, 32, 32)

        self.in_features = in_features
        self.latent_dim = latent_dim

        self.iia_weight = 0.02
        self.iia_steps = 10

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
        g_loss, _ = self.loss_to_real(fake_imgs)

        device = imgs.device
        if self.current_step % 20_000 == 0:
            # Compute IIA saliency maps usando a implementação de referência
            with torch.enable_grad():
                # Saliency map for "fake" class (class 1)
                rel_fake = compute_iia_heatmap(
                    self.discriminator,
                    fake_imgs.detach(),
                    label=1,  # fake class
                    num_steps=self.iia_steps,
                    device=device
                )

                # Saliency map for "real" class (class 0)
                rel_real = compute_iia_heatmap(
                    self.discriminator,
                    fake_imgs.detach(),
                    label=0,  # real class
                    num_steps=self.iia_steps,
                    device=device
                )

                # Saliency maps for real images
                rel_real_fake = compute_iia_heatmap(
                    self.discriminator,
                    imgs.detach(),
                    label=1,
                    num_steps=self.iia_steps,
                    device=device
                )
                rel_real_real = compute_iia_heatmap(
                    self.discriminator,
                    imgs.detach(),
                    label=0,
                    num_steps=self.iia_steps,
                    device=device
                )

            # Expand saliency maps to match image channels
            rel_fake = rel_fake.unsqueeze(1).to(device)  # [batch_size, 1, H, W]
            rel_real = rel_real.unsqueeze(1).to(device)  # [batch_size, 1, H, W]
            rel_real_fake = rel_real_fake.unsqueeze(1).to(device)
            rel_real_real = rel_real_real.unsqueeze(1).to(device)

            # Resize if needed (should already be correct size)
            if rel_fake.shape[2:] != fake_imgs.shape[2:]:
                rel_fake = F.interpolate(rel_fake, size=fake_imgs.shape[2:],
                                        mode='bicubic', align_corners=False)
                rel_real = F.interpolate(rel_real, size=fake_imgs.shape[2:],
                                        mode='bicubic', align_corners=False)
                rel_real_fake = F.interpolate(rel_real_fake, size=fake_imgs.shape[2:],
                                              mode='bicubic', align_corners=False)
                rel_real_real = F.interpolate(rel_real_real, size=fake_imgs.shape[2:],
                                              mode='bicubic', align_corners=False)

            # Apply IIA loss: encourage high values in real regions, low in fake regions
            iia_loss = (rel_fake * fake_imgs).sum() # - (rel_real * fake_imgs).sum()

            # Total generator loss
            weight_warmup = (self.current_step - 20_000) / 10_000
            split = weight_warmup * self.iia_weight
            total_loss = g_loss * (1 - split) + split * iia_loss

            image_path = pathlib.Path('output/maps/')
            image_path.mkdir(parents=True, exist_ok=True)
            grid_fake = torchvision.utils.make_grid(fake_imgs, normalize=True, scale_each=True)
            torchvision.utils.save_image(grid_fake, image_path / f'fake_imgs_{self.current_step}.png')
            grid_rel_fake = torchvision.utils.make_grid(rel_fake, normalize=True, scale_each=True)
            torchvision.utils.save_image(grid_rel_fake, image_path / f'rel_fake_{self.current_step}.png')
            grid_rel_real = torchvision.utils.make_grid(rel_real, normalize=True, scale_each=True)
            torchvision.utils.save_image(grid_rel_real, image_path / f'rel_real_{self.current_step}.png')

            grid_real = torchvision.utils.make_grid(imgs, normalize=True, scale_each=True)
            torchvision.utils.save_image(grid_real, image_path / f'real_imgs_{self.current_step}.png')
            grid_rel_real_fake = torchvision.utils.make_grid(rel_real_fake, normalize=True, scale_each=True)
            torchvision.utils.save_image(grid_rel_real_fake, image_path / f'zrel_real_fake_{self.current_step}.png')
            grid_rel_real_real = torchvision.utils.make_grid(rel_real_real, normalize=True, scale_each=True)
            torchvision.utils.save_image(grid_rel_real_real, image_path / f'zrel_real_real_{self.current_step}.png')

        return g_loss

    def compute_discriminator_loss(self, imgs: Tensor, fake_imgs: Tensor) -> Sequence[Tensor] | Tensor:
        real_loss, _ = self.loss_to_real(imgs)
        fake_loss, _ = self.loss_to_fake(fake_imgs)
        return real_loss, fake_loss
