from abc import ABCMeta, abstractmethod

from ..config import Config
from ..data_module.base import BaseDataModule


class BaseRunner(metaclass=ABCMeta):
    """Base class of Runner.

    Parameters
    ----------
    config : Config
        Configuration parameters.
    """

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def run(self, data_module: BaseDataModule):
        raise NotImplementedError

    @abstractmethod
    def save(self, output_dir: str):
        raise NotImplementedError
