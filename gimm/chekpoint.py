import pathlib
from dataclasses import dataclass

import torch

from gimm.models.definition import ModuleGAN



# TODO: Implement model_ema save
# TODO: Implement amp_scaler save
class Checkpoint:
    def __init__(self,
        model: ModuleGAN,
        optimizer_generator: torch.optim.Optimizer,
        optimizer_discriminator: torch.optim.Optimizer,
        args: dict = None,
        configs: dict = None,
        checkpoint_prefix: str = 'checkpoint-',
        checkpoint_dir: str = '',
        max_keep: int = 10,
        raise_if_dir_not_empty: bool = True,
    ):
        self.model = model
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.args = args
        self.configs = configs
        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint_dir = checkpoint_dir
        self.max_keep = max_keep
        self.extension = '.pth.tar'
        self.raise_if_dir_not_empty = raise_if_dir_not_empty

        # Create checkpoint directory
        pathlib.Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Ensure the checkpoint directory is empty if not resuming from a checkpoint
        if self.raise_if_dir_not_empty and any(pathlib.Path(self.checkpoint_dir).iterdir()):
            raise ValueError(f"Checkpoint directory {self.checkpoint_dir} is not empty and resume_checkpoint is not set.")


    def save(self, step: int):
        save_path = self.checkpoint_dir + self.checkpoint_prefix + f'{step:_}' + self.extension
        state = {
            'step': step,
            'arch': type(self.model).__name__.lower(),
            'args': self.args,
            'configs': self.configs,
            'model_state_dict': unwrap_model(self.model).state_dict(),
            'optimizer_g': self.optimizer_generator.state_dict(),
            'optimizer_d': self.optimizer_discriminator.state_dict(),
        }
        torch.save(state, save_path)

    def cycle_checkpoints(self, epoch):
        # TODO: Implement cycle_checkpoints
        # TODO: Implement last.pth.tar and best.pth.tar
        pass

    def load(self, checkpoint_path: str, should_resume_config: bool = False) -> int:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if not isinstance(checkpoint, dict):
            raise ValueError('Invalid Checkpoint. Checkpoint is not a dictionary')

        self.args = checkpoint['args']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_generator.load_state_dict(checkpoint['optimizer_g'])
        self.optimizer_discriminator.load_state_dict(checkpoint['optimizer_d'])

        configs = checkpoint.get('configs', {})
        if should_resume_config:
            self.configs = configs

        if not should_resume_config:
            # Check if the batch_size is compatible
            restored_full_batch_size = configs.get('batch_size', 0) * configs.get('grad_accum_steps', 1)
            full_batch_size = self.configs.get('batch_size', 0) * self.configs.get('grad_accum_steps', 1)
            if restored_full_batch_size != full_batch_size:
                raise ValueError(f"Restored batch size {restored_full_batch_size} is not compatible with the current batch size {full_batch_size}. Please set resume_config to True, or change the batch_size or grad_accum_steps in the current config.")

        return checkpoint['step']



def unwrap_model(model):
    # if isinstance(model, ModelEma):
    #     return unwrap_model(model.ema)

    if hasattr(model, 'module'):
        return unwrap_model(model.module)
    elif hasattr(model, '_orig_mod'):
        return unwrap_model(model._orig_mod)

    return model
