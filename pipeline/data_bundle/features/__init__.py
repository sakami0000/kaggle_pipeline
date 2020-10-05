from typing import Tuple, Union

import pandas as pd

from .registry import _FeatureRegistry

feature_registry = _FeatureRegistry()


def load_feature(
    feature_name: str
) -> Union[Tuple[pd.Series, pd.Series], Tuple[pd.DataFrame, pd.DataFrame]]:
    assert feature_name in feature_registry.feature_functions
    return feature_registry.feature_functions[feature_name]()
