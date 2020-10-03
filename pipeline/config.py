import yaml


class Config(dict): 
    """A dictionary class that allows for attribute-style access of values.
    """
    __setattr__ = dict.__setitem__

    def __getattr__(self, key):
        value = super().get(key)
        if isinstance(value, dict):
            return Config(value)
        return value


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
