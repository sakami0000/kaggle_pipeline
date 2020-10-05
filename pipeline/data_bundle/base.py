from abc import ABCMeta, abstractmethod
from typing import Any, Iterator

from ..config import Config


class BaseDataBundle(metaclass=ABCMeta):
    """Base class of DataBundle.

    Parameters
    ----------
    config : Config
        Configuration parameters.
    """

    def __init__(self, config: Config):
        self.config = config
