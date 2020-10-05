from typing import Iterator, Tuple, Union

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection._split import BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit

from ..features import load_feature
from ...config import Config
from ...utils import get_subclass_map


def build_fold(
    config: Config,
    train_x: pd.DataFrame,
    train_y: Union[pd.DataFrame, np.ndarray]
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    assert config.fold is not None
    assert config.fold.name is not None
    assert config.fold.params is not None

    folds = get_subclass_map([BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit])
    fold = folds[config.fold.name](**config.fold.params)
    train_group = None

    if config.fold.groups is not None:
        train_group = load_feature(config.fold.groups)
        
    return fold.split(train_x, train_y, groups=train_group)
