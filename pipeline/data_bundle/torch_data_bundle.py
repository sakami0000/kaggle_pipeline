import gc
from typing import Any, Iterator, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from .base import BaseDataBundle
from .data import build_loader
from .features import load_feature
from .folds import build_fold
from ..config import Config


class TorchDataBundle(BaseDataBundle):
    """DataBundle class of PyTorch implementation.

    Parameters
    ----------
    config : Config
        Configuration parameters.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.load_features()
        self.fold_indices = build_fold(config, self.train_x, self.train_y)
        self.test_data = build_loader(self.config, self.test_x, mode='valid')

    def load_features(self):
        assert self.config.features is not None
        assert self.config.targets is not None

        # load features
        train_features = []
        test_features = []
        for feature_name in self.config.features:
            # feature name is same as function name
            train_feature, test_feature = load_feature(feature_name)
            train_features.append(train_feature)
            test_features.append(test_feature)

        self.train_x = pd.concat(train_features, axis=1).reset_index(drop=True)
        self.test_x = pd.concat(test_features, axis=1).reset_index(drop=True)

        # load target variables
        train_targets = [
            load_feature(target_name)[0]
            for target_name in self.config.targets
        ]
        self.train_y = pd.concat(train_targets, axis=1).reset_index(drop=True)

        del train_features, test_features
        del train_feature, test_feature
        del train_targets
        gc.collect()

    def __len__(self) -> int:
        return len(self.train_y)

    def generate_folds_data(self) -> Iterator[Tuple[DataLoader]]:
        for train_idx, valid_idx in self.fold_indices:
            train_fold_x, train_fold_y = self.train_x.iloc[train_idx], self.train_y[train_idx]
            valid_fold_x, valid_fold_y = self.train_x.iloc[valid_idx], self.train_y[valid_idx]

            train_loader = build_loader(self.config, train_fold_x, mode='train', target=train_fold_y)
            valid_loader = build_loader(self.config, valid_fold_x, mode='valid', target=valid_fold_y)

            yield train_loader, valid_loader
