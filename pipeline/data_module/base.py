from abc import ABCMeta, abstractmethod
from typing import Any, Iterator

from ..config import Config
from ..utils import get_subclass_map


class BaseDataModule(metaclass=ABCMeta):
    """Base class of DataModule.

    Parameters
    ----------
    config : Config
        Configuration parameters.
    """

    def __init__(self, config: Config):
        self.config = config

    @classmethod
    def build(cls, config: Config) -> 'BaseDataModule':
        assert config.data_bundle is not None
        assert config.data_bundle.name is not None
        assert config.data_bundle.params is not None

        data_bundles = get_subclass_map(cls)
        obj = data_bundles[config.data_bundle.name]
        return obj(**config.data_bundle.params)
