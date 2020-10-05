from typing import Tuple

from torch import nn, optim
from torch.optim import lr_scheduler

from ...config import Config
from ...utils import get_subclass_map


def build_optimizer(config: Config, model: nn.Module) -> optim.Optimizer:
    assert config.optimizer is not None
    assert config.optimizer.name is not None
    assert config.optimizer.params is not None

    optimizers = get_subclass_map(optim.Optimizer)
    obj = optimizers[config.optimizer.name]
    return obj(model.parameters(), **config.optimizer.params)


def build_scheduler(
    config: Config,
    optimizer: optim.Optimizer
) -> Tuple[lr_scheduler._LRScheduler, lr_scheduler._LRScheduler]:
    """Create schedulers from configuration.

    Parameters
    ----------
    config : Config
        Configuration parameters.
    optimizer : optim.Optimizer
        Wrapped optimizer.

    Returns
    -------
    lr_scheduler._LRScheduler
        A scheduler which is updated on every epoch end.
    lr_scheduler._LRScheduler
        A scheduler which is updated on every batch end.
    """
    schedulers = get_subclass_map(lr_scheduler._LRScheduler)

    # epoch scheduler
    if config.epoch_scheduler is not None:
        assert config.epoch_scheduler.name is not None
        assert config.epoch_scheduler.params is not None
        
        obj = schedulers[config.epoch_scheduler.name]
        epoch_scheduler = obj(optimizer, **config.epoch_scheduler.params)
    
    else:
        epoch_scheduler = None
    
    # batch scheduler
    if config.batch_scheduler is not None:
        assert config.batch_scheduler.name is not None
        assert config.batch_scheduler.params is not None

        obj = schedulers[config.batch_scheduler.name]
        batch_scheduler = obj(optimizer, **config.batch_scheduler.params)
    
    else:
        batch_scheduler = None

    return epoch_scheduler, batch_scheduler
