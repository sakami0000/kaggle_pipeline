from . import features

from . import base
from ..config import Config
from ..utils import get_subclass_map


def build_data_bundle(config: Config) -> base.BaseDataBundle:
    assert config.data_bundle is not None
    assert config.data_bundle.name is not None
    assert config.data_bundle.params is not None

    data_bundles = get_subclass_map(base.BaseDataBundle)
    obj = data_bundles[config.data_bundle.name]
    return obj(**config.data_bundle.params)
