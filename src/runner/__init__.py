from . import models

from . import base
from . import optim
from . import torch_runner
from ..config import Config


def build_runner(config: Config) -> base.BaseRunner:
    assert config.runner is not None
    assert config.runner.name is not None
    assert config.runner.params is not None

    registry = {
        **base.__dict__,
        **torch_runner.__dict__
    }
    obj = registry[config.runner.name]
    return obj(**config.runner.params)
