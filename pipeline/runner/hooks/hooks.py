from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


@dataclass
class Hooks(object):
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    metric_fn: Callable[[np.ndarray, np.ndarray], float]
    post_forward_fn: Callable[[torch.Tensor], torch.Tensor]
