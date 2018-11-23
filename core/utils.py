"""
Email: autuanliu@163.com
Date: 2018/9/28
"""

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .loader import get_txt_data


class Timer():
    """计时器类"""

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


def early_stopping(val_loss, patience: int = 5, min_val_loss: float = 0.5):
    """使用 early_stopping 策略，判断是否要停止训练
    
    Args:
        val_loss (np.ndarray or list or tuple): 验证损失, 维度(patience,)
        patience (int, optional): Defaults to 5. 保持的长度，即验证损失不再提升的状态应保持 patience 个 epoch
        min_val_loss (float, optional): Defaults to 0.5. 到目前 epoch 为止的最小验证损失
    
    Returns:
        bool: 是否要停止训练
    """

    val_loss = np.array(val_loss).reshape(-1,)
    if val_loss.shape[0] == patience:
        return not np.any(val_loss - min_val_loss <= 0.)
    else:
        raise ValueError(f'val_loss.shape[1] or val_loss.shape[0] must be {patience}!')


def plot_save_gc_precent(txt_path: str, save_png_path: str, png_title: str, save_txt_path: str):
    """画图并保存 Granger Causality matrix 的百分比矩阵
    
    Args:
        txt_path (str): the path to Granger Causality matrix have been saved.
        save_png_path (str): the path to save figures.
        png_title (str): the title display as figure's title.
        save_txt_path (str): the path to save txt files.
    """

    data = get_txt_data(txt_path, delimiter=' ')
    gc_precent = get_gc_precent(data)
    plt.matshow(gc_precent)
    plt.title(png_title)
    plt.savefig(save_png_path)
    np.savetxt(save_txt_path, gc_precent)


def matshow(data: np.ndarray, xlabel: str, ylabel: str, title: str, png_name: str):
    """绘制矩阵图
    
    Args:
        data (np.ndarray): 要绘制的数据
        xlabel (str): 横向的标签
        ylabel (str): 纵向的标签
        title (str): 图像的名字
        png_name (str): 要保存的图像名字
    """
    
    fig, ax = plt.subplots()
    img = ax.imshow(data, cmap="YlGn")
    # ax.matshow(data, cmap="YlGn")
    # We want to show all ticks
    ax.set_xticks(np.arange(len(xlabel)))
    ax.set_yticks(np.arange(len(ylabel)))
    # and label them with the respective list entries
    ax.set_xticklabels(xlabel)
    ax.set_yticklabels(ylabel)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Create colorbar
    cbar = ax.figure.colorbar(img, ax=ax)
    cbar.ax.set_ylabel(title, rotation=-90, va="bottom")
    # Loop over data dimensions and create text annotations.
    if data.shape[0] < 5:
        for i in range(len(xlabel)):
            for j in range(len(ylabel)):
                ax.text(j, i, round(data[i, j], 4) if not abs(data[i, j]) < 1e-8 else '', ha="center", va="center", color="k")
    fig.tight_layout()
    plt.savefig(png_name)
