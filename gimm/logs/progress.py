import sys

from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm


class CustomProgressBar(TQDMProgressBar):
    BAR_FORMAT = "{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}],{rate_fmt} [{postfix}]"

    def init_train_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for training."""
        return Tqdm(
            ncols=250,
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            file=sys.stdout,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
        )

    def init_validation_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The train progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        return Tqdm(
            ncols=250,
            desc=self.validation_description,
            position=(2 * self.process_position + 1),
            disable=True,
            leave=True,
            file=sys.stdout,
            bar_format="{l_bar}{bar:50}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}],{rate_fmt} [{postfix}]",
        )
