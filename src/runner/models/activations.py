import math

import torch
from torch.nn import functional as F

ACT2FN = {
    **F.__dict__,
    'swish': swish,
    'gelu_new': gelu_new,
    'gelu_fast': gelu_fast,
    'mish': mish,
    'linear_act': linear_act
}


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
    Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def gelu_fast(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def mish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.tanh(torch.nn.functional.softplus(x))


def linear_act(x: torch.Tensor) -> torch.Tensor:
    return x
