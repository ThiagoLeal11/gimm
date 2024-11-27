import csv
import pathlib
from abc import ABC, abstractmethod

import wandb
from torch import tensor
from PIL import Image


# TODO: Implement log image


class Logger(ABC):
    # Log interval specified in terms of the number of images processed from the dataset
    interval: int
    last_logged: int

    def __init__(self, interval: int = 1_000):
        self.interval = interval
        self.last_logged = 0

    def start(self):
        """
        This method is called after everything on the model is initialized and the training will start.
        """
        pass

    def should_log(self, step: int) -> bool:
        return step - self.last_logged >= self.interval or step == 0

    def log(self, step: int, metrics: dict[str, any]) -> None:
        if self.should_log(step):
            self.last_logged = step
            self._log(step, metrics)

    def log_image(self, step: int, image: tensor, alt: str = None, prefix: str = None) -> None:
        pass

    @abstractmethod
    def _log(self, step: int, metrics: dict[str, any]) -> None:
        pass


class LoggerCsvFile(Logger):
    """
    LoggerCsvFile is a concrete implementation of Logger that logs metrics to a CSV file.

    Attributes:
        path (str): The file path where the CSV log will be saved.
        interval (int): The interval at which logging occurs.
        write_header (bool): Whether to write the header row in the CSV file.
    """

    def __init__(self, path: str, interval: int = 1_000, write_header: bool = True):
        super().__init__(interval)
        self.path = path
        self.write_header = write_header

    def _log(self, step: int, metrics: dict[str, any]) -> None:
        if not metrics:
            return

        file_exists = pathlib.Path(self.path).is_file()

        current_data = []
        if file_exists:
            with open(self.path, mode='r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    current_data.append(row)

        current_data.append(metrics)

        all_keys = set().union(*(d.keys() for d in current_data))
        with open(self.path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(current_data)


class LoggerConsole(Logger):
    def _log(self, step: int, metrics: dict[str, any]) -> None:
        print(f"Step {step}: {metrics}")


class LoggerImageFile(Logger):
    def __init__(self, path: str, interval: int = 1_000, img_format: str = "PNG"):
        super().__init__(interval)
        self.path = path
        self.img_format = img_format

    def _log(self, step: int, metrics: dict[str, any]) -> None:
        pass

    def log_image(self, step: int, image: tensor, alt: str = None, prefix: str = None) -> None:
        if not image:
            return

        image_path = pathlib.Path(self.path) / prefix / f"step_{step}.{self.img_format}"
        image_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.fromarray(image)
        img.save(image_path, format=self.img_format)


# TODO: Implement LoggerTQDM
# class LoggerTQDM(Logger):
#     def __init__(self, interval: int = 1_000):
#         super().__init__(interval)
#         self.pbar = None


class LoggerWandb(Logger):
    _instance = None

    # This class must be a singleton
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LoggerWandb, cls).__new__(cls)
            return cls._instance

        raise Exception("LoggerWandb is a singleton class. Only one instance is allowed.")

    def __init__(self, experiment: str, config: dict, tags: dict = None, resume_id: str = None, interval: int = 1_000):
        super().__init__(interval)
        self.wandb = None
        self.wandb_kwargs = {
            'project': experiment,
            'config': config,
            'tags': tags,
            'resume': 'must' if resume_id else None,
            'id': resume_id if resume_id else None,
        }

    def set_resume(self, resume_id: str):
        if not resume_id:
            return

        self.wandb_kwargs['resume'] = 'must'
        self.wandb_kwargs['id'] = resume_id

    def start(self):
        self.wandb = wandb.init(**self.wandb_kwargs)

    def _log(self, step: int, metrics: dict[str, any]) -> None:
        self.wandb.log(metrics, step=step)

    def log_image(self, step: int, image: tensor, alt: str = None, prefix: str = None) -> None:
        img = wandb.Image(image, caption=alt)
        self.wandb.log({f"{prefix}_image": img}, step=step)

    def get_run_id(self) -> str | None:
        if not self.wandb:
            return None

        return self.wandb.run.id


def get_wandb_run_id(loggers: list[Logger]) -> str | None:
    for logger in loggers:
        if isinstance(logger, LoggerWandb):
            return logger.get_run_id()

    return None
