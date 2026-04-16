import pathlib

import torch

from gimm.models.definition import ModuleGAN


# TODO: Implement model_ema save
# TODO: Implement amp_scaler save
class Checkpoint:
    def __init__(self,
        model: ModuleGAN,
        optimizer_generator: torch.optim.Optimizer | dict[str, torch.optim.Optimizer],
        optimizer_discriminator: torch.optim.Optimizer | dict[str, torch.optim.Optimizer],
        args: dict = None,
        configs: dict = None,
        checkpoint_prefix: str = 'checkpoint-',
        max_keep: int = 10,
        raise_if_dir_not_empty: bool = True,
        clean_checkpoint_dir: bool = False,
    ):
        self.model = model
        self.optimizer_generator = optimizer_generator
        self.optimizer_discriminator = optimizer_discriminator
        self.args = args
        self.configs = configs
        self.checkpoint_prefix = checkpoint_prefix
        self.max_keep = max_keep

        path = checkpoint_prefix.rsplit('/', 1)[0] if '/' in checkpoint_prefix else ''
        if path:
            # Create a checkpoint directory
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

            # Clean Checkpoint Directory
            if clean_checkpoint_dir:
                 clean_dir_deep(pathlib.Path(path))

            # Ensure the checkpoint directory is empty if not resuming from a checkpoint
            if raise_if_dir_not_empty and any(pathlib.Path(path).iterdir()):
                error_message = f"Checkpoint directory {path} is not empty and resume_checkpoint is not set."
                raise ValueError(error_message)

    def save(self, step: int):
        save_path = self.checkpoint_prefix + f'{step:_}.pth.tar'
        state = {
            'step': step,
            'arch': type(self.model).__name__.lower(),
            'args': self.args,
            'configs': self.configs,
            'model_state_dict': unwrap_model(self.model).state_dict(),
            'optimizer_g': self._get_optimizer_state(self.optimizer_generator),
            'optimizer_d': self._get_optimizer_state(self.optimizer_discriminator),
        }
        torch.save(state, save_path)

        # save_path = self.checkpoint_prefix + f'{step:_}'
        #
        # # Save model
        # torch.save({
        #     'arch': type(self.model).__name__.lower(),
        #     'args': self.args,
        #     'configs': self.configs,
        #     'model_state_dict': unwrap_model(self.model).state_dict(),
        # }, save_path + '_model.pth.tar')
        #
        # # Save Optimizers
        # torch.save({
        #     'step': step,
        #     'optimizer_g': self.optimizer_generator.state_dict(),
        #     'optimizer_d': self.optimizer_discriminator.state_dict(),
        # }, save_path + '_optimizers.pth.tar')

    def cycle_checkpoints(self, epoch):
        # TODO: Implement cycle_checkpoints
        # TODO: Implement last.pth.tar and best.pth.tar
        pass

    def load(self, checkpoint_path: str, should_resume_config: bool = False, weights_only: bool = False) -> int:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=weights_only)
        if not isinstance(checkpoint, dict):
            raise ValueError('Invalid Checkpoint. Checkpoint is not a dictionary')

        self.args = checkpoint['args']
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if not weights_only:
            self._load_optimizer_state(self.optimizer_generator, checkpoint['optimizer_g'])
            self._load_optimizer_state(self.optimizer_discriminator, checkpoint['optimizer_d'])

        configs = checkpoint.get('configs', {})
        if should_resume_config:
            self.configs = configs
        else:
            # Check if the batch_size is compatible
            restored_full_batch_size = configs.get('batch_size', 0) * configs.get('grad_accum_steps', 1)
            full_batch_size = self.configs.get('batch_size', 0) * self.configs.get('grad_accum_steps', 1)
            if restored_full_batch_size != full_batch_size:
                raise ValueError(f"Restored batch size {restored_full_batch_size} is not compatible with the current batch size {full_batch_size}. Please set resume_config to True, or change the batch_size or grad_accum_steps in the current config.")

        return checkpoint['step']

    def _get_optimizer_state(self, optimizer):
        if isinstance(optimizer, dict):
            return {k: v.state_dict() for k, v in optimizer.items()}
        return optimizer.state_dict()

    def _load_optimizer_state(self, optimizer, state):
        if isinstance(optimizer, dict):
            for k, v in optimizer.items():
                v.load_state_dict(state[k])
        else:
            optimizer.load_state_dict(state)


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
