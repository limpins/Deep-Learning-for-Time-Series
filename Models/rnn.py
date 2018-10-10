"""
Email: autuanliu@163.com
Date: 2018/9/28
Ref: 
1. [Estimating Brain Connectivity With Varying-Length Time Lags Using a Recurrent Neural Network](https://ieeexplore.ieee.org/document/8370751/)
2. [RNN-GC](https://github.com/shaozhefeng/RNN-GC)
"""

import torch
import torch.nn.functional as F
from torch import nn


class LSTM_NUE(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

    def forward(self, x):
        pass


class GRU_NUE(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

    def forward(self, x):
        pass
