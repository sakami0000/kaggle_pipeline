import torch


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return x.sigmoid()


def softmax(x: torch.Tensor) -> torch.Tensor:
    return x.softmax(dim=-1)
