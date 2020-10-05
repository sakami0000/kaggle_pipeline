from . import hooks
from . import models
from . import optim

from . import base
from . import torch_runner
from ..config import Config
from ..utils import get_subclass_map


def build_runner(config: Config) -> base.BaseRunner:
    assert config.runner is not None
    assert config.runner.name is not None
    assert config.runner.params is not None

    runners = get_subclass_map(base.BaseRunner)
    obj = runners[config.runner.name]
    return obj(**config.runner.params)
