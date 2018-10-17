"""
Email: autuanliu@163.com
Date: 2018/10/13
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
    deno = []
    im_IS = {}
    for i in range(num_channel):
        print(f'signal {i}')
        CS = list(range(num_channel))
        IS = []
        err_min_mse = 0
        while len(CS) > 0:
            err_min = 1e7
            err_min_all = None
            x_min = []
            for t in CS:
                S_tmp = IS
                S_tmp += [t]
                train_loader, test_loader = make_loader(seqdata_all, S_tmp, i, split=0.7, seq_len=20, bt_sz=32)
                x, y = MakeSeqData(seqdata_all, S_tmp, i, seq_length=20).get_tensor_data()

                err_tmp = train_valid(len(S_tmp), 15, 1, f'checkpoints/with_NUE/{signal_type}_model_weights_all{i}.pth', x, y.view(-1, 1), train_loader, test_loader)
                err_tmp_mse = F.mse_loss(err_tmp, y.float().to(device)).item()
                if err_tmp_mse < err_min:
                    err_min = err_tmp_mse
                    err_min_all = err_tmp
                    x_min = [t]
                print(f'{t} is over')
            
            deno += [err_min_all]
            IS = IS + x_min
            CS = list(set(CS) - set(x_min))
            err_min_mse = err_min

            rel_err = abs((err_min_mse - err_min)/err_min_mse)
            print(rel_err)
            if rel_err < beta:
                break
        im_IS[i] = IS
    print(im_IS)
    # end of NUE

    # 计算 err_cond
    err_cond = np.zeros((num_channel, num_channel))
    for j in range(num_channel):
        for i in range(num_channel):
            if i not in im_IS[j]:
                err_cond[i, :, j] = deno[:, j]
            else:
                train_loader, test_loader = make_loader(seqdata_all, im_IS[j], j, split=0.7, seq_len=20, bt_sz=32)
                x, y = MakeSeqData(seqdata_all, im_IS[j], j, seq_length=20).get_tensor_data()
                err_cond[i, :, j] = train_valid(len(im_IS[j]), 15, 1, f'checkpoints/with_NUE/{signal_type}_model_weights_cond{j}.pth', x, y.view(-1, 1), train_loader, test_loader).t_()

    return get_Granger_Causality(torch.from_numpy(err_cond).to(device), deno)


if __name__ == '__main__':
    # 基本设置
    timer = Timer()
    timer.start()
    bt_sz = 32
    num_epoch = 1
    num_channel = 5
    seq_len = 20
    num_trial = 1
    threshold = 0.05
    beta = 0.05  # beta in [0, 1], 95%
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

        # 保存结果
        np.savetxt(f'checkpoints/{signal_type}_granger_matrix.txt', avg_gc_matrix)
        plt.savefig(f'images/{signal_type}.png')

    # 计时结束
    timer.stop()
