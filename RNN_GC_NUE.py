"""
Email: autuanliu@163.com
Date: 2018/10/13
"""

import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from core import (MakeSeqData, Timer, get_Granger_Causality, get_mat_data,
                  make_loader, set_device, train_test_split)
from models import Modeler, RNN_Net


def train_valid(in_dim, hidden_dim, out_dim, ckpt, x, y, train_loader, test_loader):
    """训练与验证模型，每个epoch都进行训练与验证"""

    net = RNN_Net(in_dim, hidden_dim, out_dim, rnn_type='LSTM', num_layers=1)  # 创建模型实例
    opt = optim.RMSprop(net.parameters(), lr=1e-3, momentum=0.9)  # 优化器定义
    criterion = nn.MSELoss()  # 损失函数定义，由于是回归预测，所以设为 MSE loss
    model = Modeler(net, opt, criterion, device, batchsize=bt_sz)

    val_loss = []
    min_val_loss = 0.5
    for epoch in range(num_epoch):
        train_loss = model.train_model(train_loader)   # 当前 epoch 的训练损失
        test_loss = model.evaluate_model(test_loader)  # 当前 epoch 的验证损失

        # 增加 early_stopping 策略
        if test_loss <= min_val_loss:
            min_val_loss = test_loss
            model.save_trained_model(ckpt)    # 保存最好模型
        print(f'[{epoch+1}/{num_epoch}] ===>> train_loss: {train_loss: .4f} test_loss: {test_loss: .4f}')

    # 预测并计算误差
    model.load_best_model(ckpt)   # 使用最好的模型进行预测
    err = model.predit_point_by_point(x, y)[1]
    return err


def main():
    """RNN_GC 算法的实现，对应论文中的算法1(返回格兰杰矩阵)"""

    seqdata_all = get_mat_data(f'Data/{signal_type}.mat', f'{signal_type}')   # 读取数据

    # 使用NUE策略训练模型
    err_all = []
    im_IS = {}
    for k in range(num_channel):
        # 求当前信号的预测误差
        channel_set = list(range(num_channel))   # 每个信号都有可能由 0~channel 信号所影响
        input_set = []  # 当前信号的输入集合
        last_error = 1.5  # 当前信号的上一次(添加当前信号之前)预测误差, 初始化不能为 0
        min_err_all = 0

        for i in range(num_channel):
            # 这里相当于一个搜索的过程(总共选择num_channel 次)
            min_error = 1.     # 最小的误差值, 不能为0且初始状态为min_error<last_error, 要设置大一点
            min_idx = 0       # 最小误差出现的信号

            for x_idx in channel_set:
                tmp_set = copy.copy(input_set)
                tmp_set.append(x_idx)
                train_loader, test_loader = make_loader(seqdata_all, tmp_set, k, split=0.7, seq_len=20, bt_sz=32)
                x, y = MakeSeqData(seqdata_all, tmp_set, k, seq_length=20).get_tensor_data()
                err_tmp = train_valid(len(tmp_set), 15, 1, f'checkpoints/with_NUE/{signal_type}_model_weights_{x_idx}.pth', x, y, train_loader, test_loader)
                tmp_error = F.mse_loss(err_tmp.view_as(y), y.float().to(device)).item()

                # 求最小error和其x_idx
                if tmp_error < min_error:
                    min_error = tmp_error
                    min_idx = x_idx
                    min_err_all = err_tmp

            # 停止选择的条件
            if i != 0 and (np.abs(last_error - min_error) / last_error < beta or last_error < min_error):
                print('break!!!')
                break
            input_set.append(min_idx)
            channel_set.remove(min_idx)
            last_error = min_error
        
        err_all.append(min_err_all)
        im_IS[k] = input_set
        print(f'the {k} model of all input is {input_set}')
    err_all = torch.stack(err_all).squeeze()
    # end of NUE

    # 计算 err_cond
    err_cond = err_all.new_zeros(num_channel, err_all.size(1), num_channel)
    for j in range(num_channel):
        for i in range(num_channel):
            if i not in im_IS[j]:
                err_cond[i, :, j] = err_all[j, :].squeeze()
            else:
                train_loader, test_loader = make_loader(seqdata_all, im_IS[j], j, split=0.7, seq_len=20, bt_sz=32)
                x, y = MakeSeqData(seqdata_all, im_IS[j], j, seq_length=20).get_tensor_data()
                err_cond[i, :, j] = train_valid(len(im_IS[j]), 15, 1, f'checkpoints/with_NUE/{signal_type}_model_weights_cond{j}.pth', x, y, train_loader, test_loader).squeeze()

    return get_Granger_Causality(err_cond, err_all.t())


if __name__ == '__main__':
    # 基本设置
    timer = Timer()
    timer.start()
    bt_sz = 32
    num_epoch = 30
    num_channel = 5
    seq_len = 20
    num_trial = 1
    threshold = 0.05
    beta = 0.02  # beta in [0, 1], 98%
    device = set_device()
    all_signal_type = ['linear_signals', 'nonlinear_signals', 'longlag_nonlinear_signals']

    # RNN_GC
    avg_gc_matrix = 0
    for signal_type in all_signal_type:
        print(f'signal type: {signal_type}')
        for _ in range(num_trial):
            avg_gc_matrix += main()
        avg_gc_matrix /= num_trial
        avg_gc_matrix[avg_gc_matrix < threshold] = 0.  # 阈值处理
        plt.matshow(avg_gc_matrix)
        plt.title(f'{signal_type} Granger Causality Matrix')

        # 保存结果
        np.savetxt(f'checkpoints/with_NUE/{signal_type}_granger_matrix.txt', avg_gc_matrix)
        plt.savefig(f'images/with_NUE/{signal_type}.png')

    # 计时结束
    timer.stop()
