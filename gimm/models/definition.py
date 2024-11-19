from abc import abstractmethod, ABC

from torch import Tensor
from torch import nn


class ModuleGAN(ABC, nn.Module):

    @abstractmethod
    def generator_loss(self, imgs: Tensor) -> tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def discriminator_loss(self, imgs: Tensor, fake_imgs: Tensor) -> Tensor:
        pass
