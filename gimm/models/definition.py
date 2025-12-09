from abc import abstractmethod, ABC
from typing import Sequence, Optional

import torch
from torch import Tensor
from torch import nn


Size = Sequence[int]
SampleTensor = Tensor
Loss = Tensor
Logits = Tensor


class ModuleGAN(ABC, nn.Module):
    generator: nn.Module
    discriminator: nn.Module
    in_features: Size

    current_step: int
    _cached_generation: Optional[SampleTensor]

    def __init__(self):
        super(ModuleGAN, self).__init__()
        self.current_step = 0
        self._cached_generation = None

    def forward(self, latent: Tensor) -> SampleTensor:
        return self.generate(latent)

    def generate(self, latent: Tensor) -> SampleTensor:
        """
        Generate a batch of images from a latent vector
        """
        return self.generator(latent)

    def discriminate(self, samples: SampleTensor) -> Logits:
        """
        Discriminate a batch of images
        """
        return self.discriminator(samples)

    @abstractmethod
    def get_latent(self, batch_size: int) -> Tensor:
        """
        Get the latent vector from an input tensor
        """
        pass

    def generate_random_samples(self, samples: Tensor) -> SampleTensor:
        """
        Generate a batch of images from a latent vector
        """
        latent = self.get_latent(samples.shape[0]).type_as(samples)
        output = self.forward(latent)
        self._cached_generation = output.detach()
        return output

    @abstractmethod
    def compute_loss(self, imgs: Tensor, labels: Tensor) -> tuple[Loss, Logits]:
        """
        Compute the loss of the model passing the labels into the discriminator
        """
        pass

    def loss_to_real(self, imgs: Tensor) -> tuple[Loss, Logits]:
        bs = imgs.size(0)
        return self.compute_loss(
            imgs, torch.full((bs,), 1, dtype=imgs.dtype, device=imgs.device)
        )

    def loss_to_fake(self, imgs: Tensor) -> tuple[Loss, Logits]:
        bs = imgs.size(0)
        return self.compute_loss(
            imgs, torch.full((bs,), 0, dtype=imgs.dtype, device=imgs.device)
        )

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

    def get_previous_generated_samples(self, imgs: Tensor) -> SampleTensor:
        """
        Returns the previous generated images from generator or generates new ones.
        """
        # Cache miss
        if self._cached_generation is None:
            return self.generate_random_samples(imgs).detach()

        output = self._cached_generation
        self._cached_generation = None
        return output

    def before_generator_step(self):
        """
        Hook to be called before the generator optimizer.step()
        """
        pass

    def after_generator_step(self):
        """
        Hook to be called after the generator optimizer.step()
        """
        pass

    def before_discriminator_step(self):
        """
        Hook to be called before the discriminator optimizer.step()
        """
        pass

    def after_discriminator_step(self):
        """
        Hook to be called after the discriminator optimizer.step()
        """
        pass
