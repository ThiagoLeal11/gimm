from typing import Optional

import torch
from torch import Tensor

from gimm.models.definition import ModuleGAN, Loss, ImageTensor, Logits, Size
from gimm.models.vitgan.discriminator import Discriminator
from gimm.models.vitgan.generator import Generator

model_config = {
    "batch_size": 128,
    "lattent_space_size": 1024,
    "seed": 0,
    "generator_params": {
        "img_size": 32,
        "n_channels": 3,
        "lattent_size": 1024,
        "feature_hidden_size": 384,
        "n_transformer_layers": 4,
        "output_hidden_dim": 768,
        "mapping_mlp_params": {
            "layers": [],
            "activation": "gelu",
            "dropout_rate": 0.0,
        },
        "transformer_params": {
            "n_head": 4,
            "attention_dropout_rate": 0.2,
            "mlp_layers": [],
            "mlp_activation": "relu",
            "mlp_dropout": 0.2,
        },
    },
    "discriminator_params": {
        "img_size": 32,
        "n_channels": 3,
        "lattent_size": 1024,
        "n_transformer_layers": 4,
        "output_size": 1,
        "encoder_params": {
            "patch_size": 8,
            "overlap": 2,
            "dropout_rate": 0.0,
        },
        "transformer_params": {
            "n_head": 4,
            "attention_dropout_rate": 0.2,
            "mlp_layers": [],
            "mlp_activation": "relu",
            "mlp_dropout": 0.2,
        },
        "mapping_mlp_params": {
            "layers": [],
            "activation": "gelu",
            "dropout_rate": 0.0,
        },
    },
}


class VitGAN(ModuleGAN):
    def __init__(self, in_features: Optional[Size] = None):
        super().__init__()
        self.in_features = in_features
        self.criterion = torch.nn.BCELoss(reduction='mean')

    # TODO: rename to load
    def construct(self, in_features: Size) -> "ModuleGAN":
        in_features = in_features or self.in_features
        assert in_features is not None, "in_features must be provided"

        self.generator = Generator(**model_config["generator_params"])
        self.discriminator = Discriminator(**model_config["discriminator_params"])
        return self

    def get_latent(self, batch_size: int) -> Tensor:
        return torch.randn(batch_size, model_config["lattent_space_size"])

    def generate_random_images(self, x: Tensor) -> ImageTensor:
        z = self.get_latent(x.shape[0]).type_as(x)
        fake_imgs = self.generator(z)
        return fake_imgs

    def compute_loss(self, imgs: Tensor, labels: Tensor) -> tuple[Loss, Logits]:
        logits = self.discriminator(imgs).view(-1)
        return self.criterion(logits, labels), logits

# TODO: evitar retornar as imagens, fazer isso num buffer interno que chama a chacheia a ultima geração do generate_images
    def compute_generator_loss(self, imgs: Tensor) -> tuple[Tensor, Tensor]:
        fake_imgs = self.generate_random_images(imgs)
        g_loss, _ = self.loss_to_real(fake_imgs)
        return g_loss, fake_imgs.detach()

    def compute_discriminator_loss(self, imgs: Tensor, fake_imgs: Tensor) -> Tensor:
        real_loss, _ = self.loss_to_real(imgs)
        fake_loss, _ = self.loss_to_fake(fake_imgs.detach())
        return real_loss + fake_loss
