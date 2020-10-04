import inspect
from typing import Callable

import numpy as np
import sklearn.metrics
import torch
import torch.nn.functional as F

from . import post_forward_hook


class _RunnerRegistry(object):
    """Stores training functions.
    The function registered here can be used by specifying in the config file.
    This class is already instantiated as "runner_registry"
    and all functions are read from this object.

    See Also
    --------
    pipeline.data_bundle.features.registry._FeatureRegistry

    Examples
    --------
    1. Define function and register it in "runner_registry".

        >>> from pipeline import runner_registry
        >>>
        >>> @runner_registry.register_loss
        ... def mse_loss(pred, target):
        ...     return torch.mean((target - pred) ** 2)

    2. Specify the defined function in cofiguration file.

        >>> loss:
        >>>   name: mse_loss
    """

    def __init__(self):
        self.loss_functions = dict(
            inspect.getmembers(F, inspect.isfunction)
        )
        self.metric_functions = dict(
            inspect.getmembers(sklearn.metrics, inspect.isfunction)
        )
        self.post_forward_functions = dict(
            inspect.getmembers(post_forward_hook, inspect.isfunction)
        )

    def register_loss(self, function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self.loss_functions[function.__name__] = function

    def register_metric(self, function: Callable[[np.ndarray, np.ndarray], float]):
        self.metric_functions[function.__name__] = function

    def register_post_forward(self, function: Callable[[torch.Tensor], torch.Tensor]):
        self.post_forward_functions[function.__name__] = function

    def register(self, function: Callable):
        """Register a function.

        Parameters
        ----------
        function : Callable
            The function to be registered.
        """
        self.register_loss(function)
        self.register_metric(function)
        self.register_post_forward(function)
