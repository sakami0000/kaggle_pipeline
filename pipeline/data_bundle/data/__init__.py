from typing import Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from ...config import Config
from ...utils import get_subclass_map


def build_dataset(config: Config, data: Union[pd.DataFrame, np.ndarray], **kwargs) -> Dataset:
    assert config.dataset is not None
    assert config.dataset.name is not None

    if config.dataset.params is None:
        config.dataset.params = {}
    
    datasets = get_subclass_map(Dataset)
    obj = datasets[config.dataset.name]
    return obj(data, **config.dataset.params, **kwargs)


def build_loader(config: Config, data: Union[pd.DataFrame, np.ndarray], mode: str = 'train', **kwargs) -> DataLoader:
    assert mode in ['train', 'valid']
    assert config.data_loader is not None
    assert config.data_loader.train_params is not None
    assert config.data_loader.valid_params is not None

    dataset = build_dataset(config, data, **kwargs)
    data_loader = DataLoader(dataset, **config.data_loader[f'{mode}_params'])
    return data_loader
