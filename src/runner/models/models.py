import torch
from torch import nn
from torch.nn import functional as F

from .activations import ACT2FN


class SimpleNN(nn.Module):

    def __init__(self,
                 num_features: int,
                 num_targets: int,
                 hidden_size: int = 2048,
                 dropout_p: float = 0.25,
                 activation: str = 'relu'):
        super().__init__()
        self.activation = ACT2FN[activation]

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
        
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout_p)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))
        
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_p)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))
    
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = self.dense1(x)
        x = self.activation(x)
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        x = self.activation(x)
        
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        
        return x
