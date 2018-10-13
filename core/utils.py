"""
Email: autuanliu@163.com
Date: 2018/9/28
"""

import datetime as dt

import torch
import numpy as np
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
        print(f'Time taken: {(end_dt - self.start_dt).total_seconds():.2f}s')


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


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_Granger_Causality(err_cond, err_all):
    """计算 Granger Causality matrix. (err_cond, err_all 应该有相同的数据形式)
    
    Args:
        err_cond (matrix like data, numpy.ndarray or torch.Tensor): 条件误差, num_channel * n_point * num_channel
        err_all (matrix like data, numpy.ndarray or torch.Tensor): 整体误差, n_point * num_channel
    
    Returns:
        (np.ndarray) Granger Causality matrix.
    """

    if isinstance(err_cond, np.ndarray) and isinstance(err_all, np.ndarray):
        gc_matrix = np.double(err_cond).var(1) / np.double(err_all).var(0)
        gc_matrix = np.log(gc_matrix.clip(min=1.))
    elif isinstance(err_cond, torch.Tensor) and isinstance(err_all, torch.Tensor):
        gc_matrix = err_cond.double().var(1) / err_all.double().var(0)
        gc_matrix = gc_matrix.clamp(min=1.).log().cpu().numpy()
    else:
        raise ValueError('input variables should have the same type(numpy.ndarray or torch.tensor).')
    
    np.fill_diagonal(gc_matrix, 0.)   # 不考虑自身影响, 对角线为 0.
    return gc_matrix


def get_gc_precent(gc_matrix):
    """获取 Granger Causality matrix 的百分比矩阵(当前 i 信号对 j 信号影响的百分比)
    
    Args:
        gc_matrix (np.ndarray): Granger Causality matrix.
    """

    deno = np.sum(gc_matrix, axis=0)
    deno[deno == np.zeros(1)] = np.nan
    gc_precent = gc_matrix / deno
    gc_precent[np.isnan(gc_precent)] = 0.
    return gc_precent
