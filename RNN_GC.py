"""
Email: autuanliu@163.com
Date: 2018/10/11
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from core import MakeSeqData, Timer, get_mat_data, set_device, train_test_split, get_Granger_Causality, make_loader
from models import Modeler, RNN_Net


def train_valid(in_dim, hidden_dim, out_dim, ckpt, x, y, train_loader, test_loader):
    """训练与验证模型，每个epoch都进行训练与验证"""

    net = RNN_Net(in_dim, hidden_dim, out_dim, rnn_type='LSTM', num_layers=1)  # 创建模型实例
    opt = optim.RMSprop(net.parameters(), lr=1e-3, momentum=0.9)  # 优化器定义
    criterion = nn.MSELoss()  # 损失函数定义，由于是回归预测，所以设为 MSE loss
    model = Modeler(net, opt, criterion, device, batchsize=bt_sz)

    for epoch in range(num_epoch):
        train_loss = model.train_model(train_loader)   # 当前 epoch 的训练损失
        test_loss = model.evaluate_model(test_loader)  # 当前 epoch 的验证损失
        print(f'[{epoch+1}/{num_epoch}] ===>> train_loss: {train_loss: .4f} test_loss: {test_loss: .4f}')

    # 保存训练好的模型
    model.save_trained_model(ckpt)

    # 预测并计算误差
    err = model.predit_point_by_point(x, y)[1]
    return err


def main():
    """RNN_GC 算法的实现，对应论文中的算法1(返回格兰杰矩阵)"""

    seqdata_all = get_mat_data(f'Data/{signal_type}.mat', f'{signal_type}')   # 读取数据

    # 在完整数据集上训练模型
    train_loader, test_loader = make_loader(seqdata_all, split=0.7, seq_len=20, bt_sz=32)
    x_all, y_all = MakeSeqData(seqdata_all, seq_length=20, scaler_type = 'MinMaxScaler').get_tensor_data()
    err_all = train_valid(5, 15, 5, f'checkpoints/{signal_type}_model_weights.pth', x_all, y_all, train_loader, test_loader)

    # 去掉一个变量训练模型
    temp = []
    for ch in range(num_channel):
        idx = list(set(range(num_channel)) - {ch})   # 剩余变量的索引
        seq_data = seqdata_all[:, idx]   # 当前的序列数据
        train_loader, test_loader = make_loader(seq_data, split=0.7, seq_len=20, bt_sz=32)
        x, y = MakeSeqData(seq_data, seq_length=20).get_tensor_data()
        err = train_valid(4, 15, 4, f'checkpoints/{signal_type}_model_weights{ch}.pth', x, y, train_loader, test_loader)
        temp += [err]
    temp = torch.stack(temp)   # num_channel * num_point * out_dim

    # 扩充对角线

    err_cond = temp.new_zeros(temp.size(0), temp.size(1), num_channel)
    for idx in range(num_channel):
        col = list(set(range(num_channel)) - {idx})
        err_cond[idx, :, col] = temp[idx]
    return get_Granger_Causality(err_cond, err_all)


if __name__ == '__main__':
    # 基本设置
    timer = Timer()
    timer.start()
    bt_sz = 32
    num_epoch = 30
    num_channel = 5
    seq_len = 20
    num_trial = 5
    threshold = 0.05
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
        plt.title(f'{signal_type} Granger_Causality Matrix')

        # 保存结果
        np.savetxt(f'checkpoints/{signal_type}_granger_matrix.txt', avg_gc_matrix)
        plt.savefig(f'images/{signal_type}.png')

    # 计时结束
    timer.stop()
