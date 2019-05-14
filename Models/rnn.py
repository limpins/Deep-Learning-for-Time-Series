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
        batchsize (int): batchsize
        rnn_type (str): RNN网络的类型，默认为 LSTM
        num_layers (int): 隐藏层的层数，默认为 1
        dropout (float): dropout概率值，默认为 0.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, batchsize=64, rnn_type='LSTM', num_layers=1, dropout=0., bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.bidirectional = 2 if bidirectional else 1
        dropout = 0. if num_layers == 1 else dropout
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        else:
            raise ValueError("""An invalid option was supplied, options are ['LSTM', 'GRU']""")
        # way1: 使用激活函数
        # self.fc = nn.Linear(hidden_dim * self.bidirectional, hidden_dim // 2)
        # self.ac = nn.Tanh()    # 注意预测区间是 [-1, 1] 激活函数应该选择 tanh, LSTM 输出门本身就做了 tanh 激活
        # self.fc1 = nn.Linear(hidden_dim // 2, output_dim)

        # way2: 不使用激活函数
        self.fc = nn.Linear(hidden_dim * self.bidirectional, output_dim)
        self.bn = BatchNorm1dFlat(hidden_dim * self.bidirectional)
        # self.hidden = self.initHidden(batchsize)
        self.hidden = self.initHidden(batchsize)

    def forward(self, x):
        """网络的前向传播

        Args:
            x (tensor): 输入
        """

        # hidden = self.initHidden(x.size(0))  # 不保存每个 batch 的隐状态
        y, hidden = self.rnn(x, self.hidden)
        self.hidden = self.repackage_hidden(hidden)

        # pytorch的输入会记录所有时间点的输出，这里输出维度为 batchsize*seq_length*hidden_dim
        # 因为我们做的是预测模型也即多对一的RNN模型，所以取最后一个为输出即预测结果
        # 同时我们需要保存隐藏状态

        # way1: 使用激活函数
        # out = self.ac(self.fc(y))
        # out = self.fc1(out)

        # way2: 不使用激活函数
        y = self.bn(y)    # BN
        y = self.fc(y)
        return y[:, -1, :]

    def initHidden(self, batchsize):
        """初始化RNN的隐变量

        Args:
            batchsize (int): 输入数据的batchsize

        Returns:
            tuple: 返回初始化的隐变量
        """

        # GPU
        h0 = torch.zeros(self.num_layers * self.bidirectional, batchsize, self.hidden_dim).cuda()
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


class BatchNorm1dFlat(nn.BatchNorm1d):
    """`nn.BatchNorm1d`, but first flattens leading dimensions"""

    def forward(self, x):
        if x.dim() == 2: return super().forward(x)
        *f, l = x.shape
        x = x.contiguous().view(-1, l)
        return super().forward(x).view(*f, l)
