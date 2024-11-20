from abc import abstractmethod, ABC

import torch
from torch import Tensor
from torch import nn


ImageTensor = Tensor
Loss = Tensor
Logits = Tensor


class ModuleGAN(ABC, nn.Module):
    @abstractmethod
    def generate_images(self, x: Tensor) -> ImageTensor:
        """
        Generate a batch of images from a latent vector
        """
        pass

    @abstractmethod
    def loss(self, imgs: Tensor, labels: Tensor) -> tuple[Loss, Logits]:
        """
        Compute the loss of the model passing the labels into the discriminator
        """
        pass

    def real_loss(self, imgs: Tensor) -> tuple[Loss, Logits]:
        return self.loss(imgs, torch.ones(imgs.size(0), 1).type_as(imgs))

    def fake_loss(self, imgs: Tensor) -> tuple[Loss, Logits]:
        return self.loss(imgs, torch.zeros(imgs.size(0), 1).type_as(imgs))

    @abstractmethod
    def generator_loss(self, imgs: Tensor) -> tuple[Loss, ImageTensor]:
        """
        Returns the loss of the generator and the generated images.
        Remember that the images could not have been detached from the graph.
        """
        pass

    @abstractmethod
    def discriminator_loss(self, imgs: Tensor, fake_imgs: Tensor) -> Loss:
        """
        Returns the loss of the discriminator
        Remember to detach the generated images from the graph.
        """
        pass
