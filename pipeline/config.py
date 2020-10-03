from dataclasses import dataclass
import logging
import yaml


@dataclass
class Config(dict): 
    """A dictionary class that allows for attribute-style access of values.
    """
    __setattr__ = dict.__setitem__

    def __post_init__(self):
        self._logger = logging.getLogger(__name__)

    def __getattr__(self, key):
        value = super().get(key)
        if isinstance(value, dict):
            return Config(value)
        return value

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger: logging.Logger):
        self._logger = logger


def load_config(config_path: str) -> Config:
    """Load config file.

    Parameters
    ----------
    config_path : str
        Path to config file.

    Returns
    -------
    Config
        Configuration parameters.
    """
    with open(config_path) as f:
        config = Config(yaml.load(f))
    return config
