from abc import ABCMeta, abstractmethod

from ..config import Config
from ..data_bundle.base import BaseDataBundle


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
    def run(self, data_bundle: BaseDataBundle):
        raise NotImplementedError

    @abstractmethod
    def save(self, output_dir: str):
        raise NotImplementedError
