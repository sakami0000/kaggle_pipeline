from torch import nn

from . import activations
from . import models
from ...config import Config


def build_model(config: Config) -> nn.Module:
    assert config.model is not None
    assert config.model.name is not None
    assert config.model.params is not None

    registry = {
        **models.__dict__
    }
    obj = registry[config.model.name]
    return obj(**config.model.params)
