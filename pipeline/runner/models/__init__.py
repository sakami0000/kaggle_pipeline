from torch import nn

from ...config import Config
from ...utils import get_subclass_map


def build_model(config: Config) -> nn.Module:
    assert config.model is not None
    assert config.model.name is not None
    assert config.model.params is not None

    models = get_subclass_map(nn.Module)
    obj = models[config.model.name]
    return obj(**config.model.params)
