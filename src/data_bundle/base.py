from abc import ABCMeta, abstractmethod
from typing import Any, Generator

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

    @abstractmethod
    def __len__(self) -> int:
        """Returns number of training samples.

        Returns
        -------
        int
            Number of training samples.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_folds_data(self) -> Generator[Any, None, None]:
        raise NotImplementedError

    @property
    @abstractmethod
    def test_data(self) -> Any:
        raise NotImplementedError
