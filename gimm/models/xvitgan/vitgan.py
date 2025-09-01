from typing import Optional

import torch
from torch import Tensor

from gimm.models.definition import ModuleGAN, Loss, ImageTensor, Logits, Size
from gimm.models.vitgan.discriminator import Discriminator
from gimm.models.vitgan.generator import Generator
from gimm.models.xvitgan.discriminator import Discriminator
from gimm.models.xvitgan.generator import Generator

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
            "patch_size": 8,  # 2x generator patch size for overlap
            "stride_size": 4,  # Same as generator patch size
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

        if not in_features:
            in_features = (3, 32, 32)
        assert len(in_features) == 3, "in_features must be a list of 3 integers (channels, height, width)"
        assert in_features[0] in [1, 3], "in_features[0] must be either 1 (grayscale) or 3 (RGB)"
        assert in_features[1] == in_features[2], "in_features[1] and in_features[2] must be equal (height and width)"
        channels, img_size = in_features[0], in_features[1]

        model_config["generator_params"]["img_size"] = img_size
        model_config["generator_params"]["n_channels"] = channels
        model_config["discriminator_params"]["img_size"] = img_size
        model_config["discriminator_params"]["n_channels"] = channels

        self.in_features = in_features
        self.generator = Generator(**model_config["generator_params"])
        self.discriminator = Discriminator(**model_config["discriminator_params"])

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
