"""
Email: autuanliu@163.com
Date: 2018/9/28
Ref: 
1. [Estimating Brain Connectivity With Varying-Length Time Lags Using a Recurrent Neural Network](https://ieeexplore.ieee.org/document/8370751/)
2. [RNN-GC](https://github.com/shaozhefeng/RNN-GC)
3. https://github.com/pytorch/examples/blob/master/word_language_model/model.py
"""

import torch
import torch.nn.functional as F
from torch import nn


class RNN_Net(nn.Module):
    """sequence数据预测的LSTM模型或者GRU模型

    Args:
        input_dim (int): 输入维度
        hidden_dim (int): 隐藏层维度
        output_dim (int): 输出维度
        rnn_type (str): RNN网络的类型，默认为 LSTM
        num_layers (int): 隐藏层的层数，默认为 1
        dropout (float): dropout概率值，默认为 0.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, rnn_type='LSTM', num_layers=1, dropout=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        dropout = 0. if num_layers == 1 else dropout
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError("""An invalid option was supplied, options are ['LSTM', 'GRU']""")
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        """网络的前向传播

        Args:
            x (tensor): 输入
            hidden (tuple): 初始化隐变量
        """

        y, hidden = self.rnn(x, hidden)
        # pytorch的输入会记录所有时间点的输出，这里输出维度为 batchsize*seq_length*hidden_dim
        # 因为我们做的是预测模型也即多对一的RNN模型，所以取最后一个为输出即预测结果
        return self.fc(y[:, -1, :]), hidden

    def initHidden(self, batchsize):
        """初始化RNN的隐变量

        Args:
            batchsize (int): 输入数据的batchsize

        Returns:
            tuple: 返回初始化的隐变量
        """

        # 获取并创建与weight属性相同的变量，数据类型，运行设备，requires_grad等
        weight = next(self.parameters())
        h0 = weight.new_zeros(self.num_layers, batchsize, self.hidden_dim)
        if self.rnn_type == 'LSTM':
            return (h0, h0)
        else:
            return h0

    def repackage_hidden(self, hn):
        """Wraps hidden states in new Tensors, to detach them from their history.

        The hn.requires_grad == False should always be True.

        Args:
            hn (tuple or torch.tensor): hidden state((hn, cn) in LSTM and (hn) in GRU).

        Returns:
            (tuple or torch.tensor): detach hidden state.
        """

        if isinstance(hn, torch.Tensor):
            return hn.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in hn)
