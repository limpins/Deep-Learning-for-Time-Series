"""
Email: autuanliu@163.com
Date: 2018/10/11
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

from core import (Timer, get_Granger_Causality, get_mat_data, get_yaml_data,
                  make_loader, set_device)
from models import Modeler, RNN_Net


def train_valid(in_dim, hidden_dim, out_dim, ckpt, test_data, loaders):
    """训练与验证模型，每个epoch都进行训练与验证"""

    net = RNN_Net(in_dim, hidden_dim, out_dim, rnn_type=cfg['rnn_type'], num_layers=cfg['num_layers'], dropout=cfg['dropout'])  # 创建模型实例
    opt = optim.RMSprop(net.parameters(), lr=cfg['lr_rate'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])  # 优化器定义
    criterion = nn.MSELoss()  # 损失函数定义，由于是回归预测，所以设为 MSE loss
    model = Modeler(net, opt, criterion, device)

    val_loss = []
    min_val_loss = 0.5
    for epoch in range(cfg['num_epoch']):
        train_loss = model.train_model(loaders['train'])   # 当前 epoch 的训练损失
        valid_loss = model.evaluate_model(loaders['valid'])  # 当前 epoch 的验证损失

        # 增加 early_stopping 策略
        # if valid_loss <= min_val_loss:
        #     min_val_loss = valid_loss
        #     model.save_trained_model(ckpt)
        print(f"[{epoch+1}/{cfg['num_epoch']}] ===>> train_loss: {train_loss: .4f} | valid_loss: {valid_loss: .4f}")

    # 预测并计算误差
    model.save_trained_model(ckpt)
    model.load_best_model(ckpt)   # 使用最好的模型进行预测
    err = model.predit_point_by_point(*test_data)[1]
    return err


def main():
    """RNN_GC with NUE(non-uniform embedding) 算法的实现，对应论文中的算法2(返回格兰杰矩阵)"""

    seqdata_all = get_mat_data(f'Data/{signal_type}.mat', f'{signal_type}')   # 读取数据

    # 在完整数据集上训练模型
    train_loader, valid_loader, test_loader = make_loader(seqdata_all, tt_split=cfg['tt_split'], tv_split=cfg['tv_split'], seq_len=cfg['seq_len'], bt_sz=cfg['bt_sz'])
    loaders = {'train': train_loader, 'valid': valid_loader}
    err_all = train_valid(5, cfg['num_hidden'], 5, f'checkpoints/without_NUE/{signal_type}_model_weights.pth', test_loader.dataset.get_tensor_data(), loaders)

    # 去掉一个变量训练模型
    temp = []
    for ch in range(num_channel):
        idx = list(set(range(num_channel)) - {ch})   # 剩余变量的索引
        seq_data = seqdata_all[:, idx]   # 当前的序列数据
        train_loader, valid_loader, test_loader = make_loader(seq_data, tt_split=cfg['tt_split'], tv_split=cfg['tv_split'], seq_len=cfg['seq_len'], bt_sz=cfg['bt_sz'])
        loaders = {'train': train_loader, 'valid': valid_loader}
        err = train_valid(4, cfg['num_hidden'], 4, f'checkpoints/without_NUE/{signal_type}_model_weights{ch}.pth', test_loader.dataset.get_tensor_data(), loaders)
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
    num_channel = 5
    config = get_yaml_data('configs/cfg.yaml')
    device = set_device()
    all_signal_type = ['linear_signals', 'nonlinear_signals', 'longlag_nonlinear_signals']

    # RNN_GC
    avg_gc_matrix = 0
    # for signal_type in all_signal_type:
    signal_type = all_signal_type[2]
    print(f'signal type: {signal_type}')
    cfg = config[signal_type]
    for _ in range(cfg['num_trial']):
        avg_gc_matrix += main()
    avg_gc_matrix /= cfg['num_trial']
    avg_gc_matrix[avg_gc_matrix < cfg['threshold']] = 0.  # 阈值处理
    plt.matshow(avg_gc_matrix)
    plt.title(f'{signal_type} Granger_Causality Matrix')

    # 保存结果
    np.savetxt(f'checkpoints/without_NUE/{signal_type}_granger_matrix.txt', avg_gc_matrix)
    plt.savefig(f'images/without_NUE/{signal_type}_Granger_Matrix.png')

    # 计时结束
    timer.stop()
