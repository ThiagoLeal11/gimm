import json
import pathlib

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from gimm.models.definition import ModuleGAN
from gimm.run.config import TrainerConfig
from gimm.scheduler.scheduler import Scheduler


# TODO: Implement model_ema save
# TODO: Implement amp_scaler save
class Checkpoint:
    MODEL_FILE_NAME = 'model.safetensors'
    TRAINING_STATE_FILE_NAME = 'training-state.pth.tar'
    METADATA_FILE_NAME = 'metadata.json'

    def __init__(self,
        model: ModuleGAN,
        optimizer_generator: dict[str, torch.optim.Optimizer],
        optimizer_discriminator: dict[str, torch.optim.Optimizer],
        scheduler_generator: dict[str, Scheduler],
        scheduler_discriminator: dict[str, Scheduler],
        args: dict = None,
        configs: TrainerConfig | None = None,
    ):
        self.model = model
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.scheduler_generator = scheduler_generator
        self.scheduler_discriminator = scheduler_discriminator
        self.args = args or {}
        self.configs = configs or TrainerConfig()
        path = self.get_checkpoint_path()
        if path:
            # Clean Checkpoint Directory
            if self.configs.delete_checkpoint:
                clean_dir_deep(pathlib.Path(path))

            # Ensure the checkpoint directory is empty if not resuming from a checkpoint
            has_some_checkpoint = self.configs.resume_checkpoint or self.configs.pretrained
            if not has_some_checkpoint and any(pathlib.Path(path).iterdir()):
                error_message = (
                    f"Checkpoint directory {path} is not empty and resume_checkpoint is not set. "
                    f"This can lead to unexpected overwrite. Please select a different path."
                )
                raise ValueError(error_message)

    def get_checkpoint_path(self) -> str:
        return str(pathlib.Path(self.configs.output_path) / 'checkpoints')

    def get_checkpoint_prefix(self) -> str:
        return str(pathlib.Path(self.get_checkpoint_path()) / 'checkpoint-')

    def save(self, step: int):
        # Create a checkpoint directory
        pathlib.Path(self.get_checkpoint_path()).mkdir(parents=True, exist_ok=True)

        save_dir = pathlib.Path(f"{self.get_checkpoint_prefix()}{step:_}")
        save_dir.mkdir(parents=True, exist_ok=True)

        training_state = {
            'args': self.args,
            'optimizer_g': self._get_optimizer_state(self.optimizer_generator),
            'optimizer_d': self._get_optimizer_state(self.optimizer_discriminator),
            'scheduler_g': self._get_scheduler_state(self.scheduler_generator),
            'scheduler_d': self._get_scheduler_state(self.scheduler_discriminator),
        }
        save_safetensors(self._cpu_state_dict(), str(save_dir / self.MODEL_FILE_NAME))
        torch.save(training_state, save_dir / self.TRAINING_STATE_FILE_NAME)

        with (save_dir / self.METADATA_FILE_NAME).open('w', encoding='utf-8') as file:
            json.dump(self._to_jsonable({
                'step': step,
                'arch': type(self.model).__name__.lower(),
                'configs': self.configs.to_dict(),
            }), file, indent=2, sort_keys=True)

    def cycle_checkpoints(self, epoch):
        # TODO: Implement cycle_checkpoints
        # TODO: Implement last.pth.tar and best.pth.tar
        pass

    def load(self, checkpoint_path: str) -> int:
        path = pathlib.Path(checkpoint_path)
        if not path.is_dir() and path.suffix != '.safetensors':
            raise ValueError(f'Invalid checkpoint. Checkpoint ({path}) is neither a directory nor a model file.')

        # Load the model
        if path.suffix == '.safetensors':
            self.model.load_state_dict(load_safetensors(str(path)))
            return 0
        self.model.load_state_dict(load_safetensors(str(path / self.MODEL_FILE_NAME)))

        training_state = torch.load(path / self.TRAINING_STATE_FILE_NAME, map_location='cpu')
        self.args = training_state.get('args', self.args)

        self._load_optimizer_state(self.optimizer_generator, training_state['optimizer_g'])
        self._load_optimizer_state(self.optimizer_discriminator, training_state['optimizer_d'])

        # Load only the schedulers that arend overrides.
        if not self._has_scheduler(self.scheduler_generator) and 'scheduler_g' in training_state:
            self._load_scheduler_state(self.scheduler_generator, training_state['scheduler_g'])
        if not self._has_scheduler(self.scheduler_discriminator) and 'scheduler_d' in training_state:
            self._load_scheduler_state(self.scheduler_discriminator, training_state['scheduler_d'])

        with (path / self.METADATA_FILE_NAME).open('r', encoding='utf-8') as file:
            metadata = json.load(file)
            training_configs = metadata['configs']

        # Keeping the user overrides to the resumed config
        override_values = self.configs.get_user_overrides()
        training_configs.update(override_values)

        # Cleanup not initialized optimizers
        if isinstance(training_configs['g_optimizer'], str):
            training_configs.pop('g_optimizer')
        if isinstance(training_configs['d_optimizer'], str):
            training_configs.pop('d_optimizer')

        self.configs.set(training_configs)
        return metadata['step'] + 1

    def _cpu_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().contiguous()
            for key, value in unwrap_model(self.model).state_dict().items()
        }

    @classmethod
    def _to_jsonable(cls, value):
        if isinstance(value, dict):
            return {str(k): cls._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_jsonable(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, pathlib.Path):
            return str(value)
        if isinstance(value, torch.device):
            return str(value)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        return repr(value)

    @staticmethod
    def _get_optimizer_state(optimizers: dict[str, torch.optim.Optimizer]):
        return {k: v.state_dict() for k, v in optimizers.items()}

    @staticmethod
    def _load_optimizer_state(optimizers: dict[str, torch.optim.Optimizer], state):
        for k, v in optimizers.items():
            v.load_state_dict(state[k])

    @staticmethod
    def _get_scheduler_state(schedulers: dict[str, Scheduler] | None):
        if schedulers is None:
            return None
        return {k: v.state_dict() for k, v in schedulers.items()}

    @staticmethod
    def _load_scheduler_state(schedulers: dict[str, Scheduler] | None, state):
        if schedulers is None or state is None:
            return

        for k, v in schedulers.items():
            if k in state:
                v.load_state_dict(state[k])

    @staticmethod
    def _has_scheduler(schedulers: dict[str, Scheduler]):
        return all(isinstance(sch, Scheduler) for sch in schedulers.values())


def unwrap_model(model):
    # if isinstance(model, ModelEma):
    #     return unwrap_model(model.ema)

    if hasattr(model, 'module'):
        return unwrap_model(model.module)
    elif hasattr(model, '_orig_mod'):
        return unwrap_model(model._orig_mod)

    return model


def clean_dir_deep(path: pathlib.Path):
    if not path.exists():
        return

    for path in path.iterdir():
        if path.is_dir():
            clean_dir_deep(path)
        else:
            path.unlink()
