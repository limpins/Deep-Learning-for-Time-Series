"""
Email: autuanliu@163.com
Date: 2018/9/28
"""

import datetime as dt

import torch
from torch import nn


class Timer():
    """计时器类
    """

    def __init__(self):
        pass

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        print(f'Time taken: {end_dt - self.start_dt}')


def set_device():
    """设置运行设备CPU或者GPU

    Returns:
        (torch.device): 设备对象
    """

    return torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')


def init_params(net):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 1)
            nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal(m.weight, std=1e-3)
            if m.bias:
                nn.init.constant(m.bias, 0)


def one_hot_encoding(labels, num_classes):
    """Embedding labels to one-hot.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    """
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]
