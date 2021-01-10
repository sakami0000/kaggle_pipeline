import numpy as np
import torch
from torch.utils.data import Dataset


class SubDataset(Dataset):
    """Subset of a dataset at specified indices.

    Parameters
    ----------
    dataset : np.ndarray, shape = (n_samples, n_features)
        The whole dataset.
    target : np.ndarray of shape (n_samples, n_targets), optional
        The target values.
    indices : np.ndarray, optional
        If specified, returns subset of the dataset at these indices.
    """

    def __init__(self,
                 dataset: np.ndarray,
                 target: np.ndarray = None,
                 indices: np.ndarray = None):
        if indices is None:
            indices = np.arange(len(dataset))

        if target is not None:
            target = torch.FloatTensor(target)

        self.dataset = torch.FloatTensor(dataset)
        self.target = target
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data = ({'x': self.dataset[self.indices[idx]]})
        if self.target is not None:
            data += (self.target[self.indices[idx]])
        return data
