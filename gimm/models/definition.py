from abc import abstractmethod, ABC
from typing import Sequence, Optional, Callable

import torch
from torch import Tensor
from torch import nn


DEFAULT_NET = 'default'


Size = Sequence[int]
SampleTensor = Tensor
Loss = Tensor
Logits = Tensor


class ModuleGAN(ABC, nn.Module):
    generators: nn.ModuleDict
    discriminators: nn.ModuleDict
    in_features: Size

    current_step: int
    _cached_generation: Optional[SampleTensor]

    @property
    def generator(self) -> nn.Module:
        if len(self.generators) == 1:
            return next(iter(self.generators.values()))

        raise RuntimeError(
            "More than one generator found in the architecture. "
            "You will need to override some functions for this custom architecture"
        )

    @property
    def discriminator(self) -> nn.Module:
        if len(self.discriminators) == 1:
            return next(iter(self.discriminators.values()))

        raise RuntimeError(
            "More than one discriminator found in the architecture. "
            "You will need to override some functions for this custom architecture"
        )

    def set_generator(self, gen: nn.Module):
        self.generators[DEFAULT_NET] = gen

    def set_discriminator(self, disc: nn.Module):
        self.discriminators[DEFAULT_NET] = disc

    def __init__(self):
        super(ModuleGAN, self).__init__()
        self.current_step = 0
        self._cached_generation = None
        self.generators = nn.ModuleDict()
        self.discriminators = nn.ModuleDict()

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
    def compute_generator_loss(self, imgs: Tensor) -> Sequence[Loss] | Loss:
        """
        Returns the loss of the generator and the generated images.
        Remember that the images could not have been detached from the graph.
        """
        pass

    @abstractmethod
    def compute_discriminator_loss(self, imgs: Tensor, fake_imgs: Tensor) -> Sequence[Loss] | Loss:
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

    def generator_train_step(self, batch: tuple[Tensor, Tensor], backward: Callable[[Loss], None]) -> Loss:
        # Freeze discriminator
        for d in self.discriminators.values():
            for p in d.parameters():
                p.requires_grad = False

        # Train generator
        imgs, labels = batch
        g_loss = self.compute_generator_loss(imgs)
        backward(g_loss)

        # Unfreeze discriminator
        for d in self.discriminators.values():
            for p in d.parameters():
                p.requires_grad = True

        return g_loss

    def discriminator_train_step(self, batch: tuple[Tensor, Tensor], backward: Callable[[Loss], None]) -> Loss:
        imgs, labels = batch
        fake_imgs = self.get_previous_generated_samples(imgs)
        d_loss = self.compute_discriminator_loss(imgs, fake_imgs)
        backward(d_loss)
        return d_loss

    def generator_optimizer_step(self, optimizers: dict[str, torch.optim.Optimizer]):
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad()

    def discriminator_optimizer_step(self, optimizers: dict[str, torch.optim.Optimizer]):
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad()

    def log_generator_loss(self, tensors: Sequence[Loss] | Loss) -> dict[str, float]:
        if isinstance(tensors, Tensor):
            return {'g_loss': tensors.item()}
        return {f'g_loss_{i}': loss.item() for i, loss in enumerate(tensors)}

    def log_discriminator_loss(self, tensors: Sequence[Loss] | Loss) -> dict[str, float]:
        if isinstance(tensors, Tensor):
            return {'d_loss': tensors.item()}
        return {f'd_loss_{i}': loss.item() for i, loss in enumerate(tensors)}