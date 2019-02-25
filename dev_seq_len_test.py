"""
Email: autuanliu@163.com
Date: 2018/10/11
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

from core import Timer, get_mat_data, make_loader, set_device
from Models import Modeler, RNN_Net


def train_valid(in_dim, hidden_dim, out_dim, test_data, loaders):
    """训练与验证模型，每个epoch都进行训练与验证
    """

    net = RNN_Net(in_dim, hidden_dim, out_dim, rnn_type='LSTM', num_layers=1, dropout=0)    # 创建模型实例
    opt = optim.RMSprop(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0)    # 优化器定义
    lr_decay2 = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)    # 学习率衰减
    criterion = nn.MSELoss()    # 损失函数定义，由于是回归预测，所以设为 MSE loss
    model = Modeler(net, opt, criterion, device)

    for epoch in range(40):
        train_loss = model.train_model(loaders['train'])    # 当前 epoch 的训练损失
        valid_loss = model.evaluate_model(loaders['valid'])    # 当前 epoch 的验证损失
        lr_decay2.step(valid_loss)
        print(f"[{epoch+1}/{100}] ===>> train_loss: {train_loss: .4f} | valid_loss: {valid_loss: .4f}")

    # 预测并计算误差
    err = model.predit_point_by_point(*test_data)[1]
    return err


def main(seq_length):
    seqdata_all = get_mat_data(f'dev_seq_data/{signal_type}.mat', f'{signal_type}')    # 读取数据

    # 在完整数据集上训练模型
    train_loader, valid_loader, test_loader = make_loader(
        seqdata_all, tt_split=0.7, tv_split=0.8, seq_len=seq_length, bt_sz=32)
    loaders = {'train': train_loader, 'valid': valid_loader}
    err_all = train_valid(2, 15, 2, test_loader.dataset.get_tensor_data(), loaders)


# 基本设置
device = set_device()

set1 = {'longlag_nonlinear_signals': [3, 5, 8, 10, 12, 15, 18, 20], 'nonlinear_signals': [3, 5, 8, 10, 13, 15, 18, 20]}
all_signal_type = ['nonlinear_signals', 'longlag_nonlinear_signals']
time_spend = {'longlag_nonlinear_signals': [], 'nonlinear_signals': []}
for signal_type in all_signal_type:
    for seq_len in set1[signal_type]:
        print(f'{signal_type}, seq length: {seq_len}')
        timer = Timer()
        timer.start()
        main(seq_len)
        # 计时结束
        b = timer.stop()
        time_spend[signal_type].append(b)

    plt.plot(set1[signal_type], time_spend[signal_type])
    plt.show()
