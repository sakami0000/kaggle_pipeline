import functools
import inspect
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import sklearn.metrics

from . import post_forward_hook
from .hooks import Hooks
from .registry import _RunnerRegistry
from ...config import Config
from ...utils import get_subclass_map

runner_registry = _RunnerRegistry()


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

    # loss function
    if config.loss.name in runner_registry.loss_functions:
        func = runner_registry.loss_functions[config.loss.name]
        if config.loss.params is not None:
            func = functools.partial(func, **config.loss.params)
        return func

    # loss class
    assert config.loss.params is not None

    loss_classes = get_subclass_map([_Loss, nn.Module])
    obj = loss_classes[config.loss.name]
    return obj(**config.loss.params)


def _build_metric(config: Config) -> Callable[[np.ndarray, np.ndarray], float]:
    assert config.metric is not None
    assert config.metric.name is not None

    func = runner_registry.metric_functions[config.metric.name]
    if config.metric.params is not None:
        func = functools.partial(func, **config.metric.params)
    return func


def _build_post_forward(config: Config) -> Callable[[torch.Tensor], torch.Tensor]:
    if config.post_forward is None:
        return post_forward_hook.identity
    
    else:
        assert config.post_forward.name is not None

        func = runner_registry.post_forward_functions[config.post_forward.name]
        if config.post_forward.params is not None:
            func = functools.partial(func, **config.post_forward.params)
        return func
