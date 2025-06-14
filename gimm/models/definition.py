from abc import abstractmethod, ABC
from typing import Sequence

import torch
from torch import Tensor
from torch import nn


Size = Sequence[int]
ImageTensor = Tensor
Loss = Tensor
Logits = Tensor


class ModuleGAN(ABC, nn.Module):
    generator: nn.Module
    discriminator: nn.Module

    @abstractmethod
    def construct(self, in_features: Size) -> 'ModuleGAN':
        return self

    def forward(self, latent: Tensor) -> ImageTensor:
        return self.generator(latent)

    def generate(self, latent: Tensor) -> ImageTensor:
        """
        Generate a batch of images from a latent vector
        """
        return self.forward(latent)

    def discriminate(self, imgs: ImageTensor) -> Logits:
        """
        Discriminate a batch of images
        """
        return self.discriminator(imgs)

    @abstractmethod
    def get_latent(self, batch_size: int) -> Tensor:
        """
        Get the latent vector from an input tensor
        """
        pass

    @abstractmethod
    def generate_random_images(self, batch_size: Tensor) -> ImageTensor:
        """
        Generate a batch of images from a latent vector
        """
        pass

    @abstractmethod
    def compute_loss(self, imgs: Tensor, labels: Tensor) -> tuple[Loss, Logits]:
        """
        Compute the loss of the model passing the labels into the discriminator
        """
        pass

    def loss_to_real(self, imgs: Tensor) -> tuple[Loss, Logits]:
        bs = imgs.size(0)
        return self.compute_loss(imgs, torch.full((bs, ), 1, dtype=imgs.dtype, device=imgs.device))

    def loss_to_fake(self, imgs: Tensor) -> tuple[Loss, Logits]:
        bs = imgs.size(0)
        return self.compute_loss(imgs, torch.full((bs, ), 0, dtype=imgs.dtype, device=imgs.device))

    @abstractmethod
    def compute_generator_loss(self, imgs: Tensor) -> tuple[Loss, ImageTensor]:
        """
        Returns the loss of the generator and the generated images.
        Remember that the images could not have been detached from the graph.
        """
        pass

    @abstractmethod
    def compute_discriminator_loss(self, imgs: Tensor, fake_imgs: Tensor) -> Loss:
        """
        Returns the loss of the discriminator
        Remember to detach the generated images from the graph.
        """
        pass
