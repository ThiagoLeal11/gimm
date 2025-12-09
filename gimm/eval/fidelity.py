import math
import pathlib
import shutil
from abc import abstractmethod

import torch
import torch_fidelity
from torch import Tensor
from torch.utils.data import DataLoader
from torch_fidelity import GenerativeModelBase

from gimm.eval.compute import EvalMetric
from gimm.models.definition import ModuleGAN


PRIMITIVE = str | int | float | bool


def _to_int8(images: torch.Tensor) -> torch.Tensor:
    uint8_images = ((images + 1) * 127.5).clamp(0, 255).to(torch.uint8)

    # Verifica se é uma imagem em escala de cinza (1 canal) e converte para RGB
    if uint8_images.ndim >= 3 and uint8_images.shape[1] == 1:
        uint8_images = uint8_images.expand(-1, 3, -1, -1)

    return uint8_images


class FidelityModelWrapper(GenerativeModelBase):
    def __init__(self, module: ModuleGAN):
        super().__init__()
        self.module = module
        self.latent_size = module.get_latent(1).shape[1]
        self.device = next(module.parameters()).device

    @property
    def z_size(self):
        return self.latent_size

    @property
    def z_type(self):
        return 'normal'

    @property
    def num_classes(self):
        return 0

    def forward(self, *args, **kwargs):
        x = args[0]
        assert isinstance(x, Tensor), "Input must be a Tensor"

        batch_size = args[0].shape[0]
        latent = self.module.get_latent(batch_size).to(self.device)
        fake_images = self.module.generate_random_samples(latent)
        quantized_images = _to_int8(fake_images)
        return quantized_images


class FidelityDataloaderWrapper(GenerativeModelBase):
    def __init__(self, dataloader: DataLoader, batch_size: int, device: torch.device):
        super().__init__()
        self.device = device
        generator = self.dataloader_to_stream(dataloader, batch_size)
        self.stream = iter(generator)

    @staticmethod
    def dataloader_to_stream(dataloader: DataLoader, batch_size: int):
        buffer = torch.Tensor([])
        while True:
            for (imgs, labels) in dataloader:
                buffer = torch.cat((buffer, imgs), dim=0)

                if buffer.shape[0] >= batch_size:
                    batch = buffer[:batch_size]
                    buffer = buffer[batch_size:]
                    yield batch

    @property
    def z_size(self):
        return 4  # Random latent vector size

    @property
    def z_type(self):
        return 'normal'

    @property
    def num_classes(self):
        return 0

    def forward(self, *args, **kwargs):
        batch = next(self.stream)
        return _to_int8(batch).to(self.device)


class FidelityEvalMetric(EvalMetric):
    def __init__(self):
        super().__init__(samples=0)

    def reset_real_distribution(self) -> None:
        raise NotImplementedError

    def reset_fake_distribution(self) -> None:
        raise NotImplementedError

    def update(self, batch: tuple[Tensor, Tensor]) -> None:
        raise NotImplementedError

    def compute(self) -> dict[str, any]:
        raise NotImplementedError

    @abstractmethod
    def compile(self) -> dict[str, PRIMITIVE]:
        pass


class FrechetInceptionDistance(FidelityEvalMetric):
    def compile(self) -> dict[str, PRIMITIVE]:
        return {
            "fid": True,
        }


class InceptionScore(FidelityEvalMetric):
    def compile(self) -> dict[str, PRIMITIVE]:
        return {
            "isc": True,
        }

class KernelInceptionDistance(FidelityEvalMetric):
    def compile(self) -> dict[str, PRIMITIVE]:
        return {
            "kid": True,
        }

class PerceptualPathLength(FidelityEvalMetric):
    def compile(self) -> dict[str, PRIMITIVE]:
        return {
            "ppl": True,
        }

class PrecisionRecall(FidelityEvalMetric):
    def compile(self) -> dict[str, PRIMITIVE]:
        return {
            "PRC": True,
        }


def clean_cache(config: dict) -> None:
    cache_path = pathlib.Path(config['output_path']) / "cache" / "fidelity_cache"
    if cache_path.exists():
        shutil.rmtree(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)


def compute_metrics(model: ModuleGAN, dataloader: DataLoader, metrics: list[FidelityEvalMetric], config: dict) -> dict[str, float]:
    kwargs = {
        "cache": True,
        "cache_root": config['output_path'] + "/cache/fidelity_cache",
        "input2_cache_name": config['experiment'],
        "input1_model_num_samples": config['samples'],
        "input2_model_num_samples": config['samples'],
        "batch_size": config['batch_size'],
        "verbose": config['verbose'],
        "cuda": config['device'].type != "cpu",
    }
    for metric in metrics:
        kwargs.update(metric.compile())

    reference_wrapper = FidelityDataloaderWrapper(dataloader, config['batch_size'], config['device'])
    generated_wrapper = FidelityModelWrapper(model)

    metrics_dict = torch_fidelity.calculate_metrics(
        input1=generated_wrapper,
        input2=reference_wrapper,
        **kwargs,
    )
    return metrics_dict
