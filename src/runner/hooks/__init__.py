import functools
from typing import Callable

import numpy as np
import torch
from torch import nn
import sklearn.metrics

from . import post_forward_hook
from .hooks import Hooks
from ...config import Config


def build_hooks(config: Config) -> Hooks:
    hooks = Hooks(
        loss_fn=_build_loss(config),
        metric_fn=_build_metric(config),
        post_forward_fn=_build_post_forward(config)
    )
    return hooks


def _build_loss(config: Config) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    assert config.loss is not None
    assert config.loss.name is not None
    assert config.loss.params is not None

    registry = {
        **nn.__dict__
    }
    obj = registry[config.loss.name]
    return obj(**config.loss.params)


def _build_metric(config: Config) -> Callable[[np.ndarray, np.ndarray], float]:
    assert config.metric is not None
    assert config.metric.name is not None
    assert config.metric.params is not None

    registry = {
        **sklearn.metrics.__dict__
    }
    func = registry[config.metric.name]
    return functools.partial(func, **config.metric.params)


def _build_post_forward(config: Config) -> Callable[[torch.Tensor], torch.Tensor]:
    if config.post_forward is None:
        return post_forward_hook.identity
    
    else:
        assert config.post_forward.name is not None
        assert config.post_forward.params is not None

        registry = {
            **post_forward_hook.__dict__
        }
        func = registry[config.post_forward.name]
        return functools.partial(func, **config.post_forward.params)
