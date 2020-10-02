from . import features

from . import base
from ..config import Config


def build_data_bundle(config: Config) -> base.BaseDataBundle:
    assert config.data_bundle is not None
    assert config.data_bundle.name is not None
    assert config.data_bundle.params is not None

    registry = {
        **base.__dict__
    }
    obj = registry[config.data_bundle.name]
    return obj(**config.data_bundle.params)
