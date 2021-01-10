from typing import Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Sampler

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


def build_sampler(config: Config, data: Union[pd.DataFrame, np.ndarray]) -> Sampler:
    samplers = get_subclass_map(Sampler)
    sampler = None
    batch_sampler = None

    # sampler
    if config.sampler is not None:
        assert config.sampler.name is not None
        assert config.sampler.params is not None
        
        sampler = samplers[config.sampler.name](data, **config.sampler.params)

    # batch sampler
    if config.batch_sampler is not None:
        assert config.batch_sampler.name is not None
        assert config.batch_sampler.params is not None

        batch_sampler = samplers[config.batch_sampler.name](data, **config.batch_sampler.params)

    return sampler, batch_sampler


def build_loader(config: Config, data: Union[pd.DataFrame, np.ndarray], mode: str = 'train', **kwargs) -> DataLoader:
    assert mode in ['train', 'valid']
    assert config.data_loader is not None
    assert config.data_loader.train_params is not None
    assert config.data_loader.valid_params is not None

    dataset = build_dataset(config, data, **kwargs)
    sampler, batch_sampler = build_sampler(config, data)
    data_loader = DataLoader(dataset,
                             sampler=sampler,
                             batch_sampler=batch_sampler,
                             **config.data_loader[f'{mode}_params'])
    return data_loader
