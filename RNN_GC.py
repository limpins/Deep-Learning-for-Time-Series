"""
Email: autuanliu@163.com
Date: 2018/10/11
"""

import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt

from Models import Modeler, RNN_Net, AdaBoundW, AdaBound
from core import (Timer, get_Granger_Causality, get_json_data, get_mat_data, make_loader, matshow, set_device, time_series_split)


def train_net(train_set, valid_set, test_set, in_dim, out_dim, cfg):
    # 在完整数据集上训练模型
    train_loader, valid_loader, test_loader = make_loader(train_set,
                                                          valid_set,
                                                          test_set,
                                                          seq_len=cfg['seq_len'],
                                                          num_shift=cfg['num_shift'],
                                                          bt_sz=cfg['bt_sz'])

    # 创建模型实例
    net = RNN_Net(in_dim,
                  cfg['hidden_dim'],
                  out_dim,
                  batchsize=cfg['bt_sz'],
                  rnn_type=cfg['rnn_type'],
                  num_layers=cfg['num_layers'],
                  dropout=cfg['dropout'],
                  bidirectional=cfg['bidirectional'])

    # 优化器定义
    # opt = optim.RMSprop(net.parameters(), lr=cfg['lr_rate'], weight_decay=cfg['weight_decay'])
    # opt = AdaBoundW(net.parameters(), lr=cfg['lr_rate'], weight_decay=cfg['weight_decay'])
    opt = AdaBound(net.parameters(), lr=cfg['lr_rate'], weight_decay=cfg['weight_decay'])

    # 损失定义
    criterion = nn.MSELoss()

    # 封装
    model = Modeler(net, opt, criterion, set_device())

    # 训练
    for epoch in range(cfg['num_epoch']):
        train_loss = model.train_model(train_loader)    # 当前 epoch 的训练损失
        valid_loss = model.evaluate_model(valid_loader)    # 当前 epoch 的验证损失
        print(f"[{epoch + 1}/{cfg['num_epoch']}] ===>> train_loss: {train_loss: .4f} | valid_loss: {valid_loss: .4f}")

    # 训练结束 预测
    prediction, err = model.predit_point_by_point(test_loader.dataset.data, test_loader.dataset.target)

    # 可视化预测效果
    # for ch in range(prediction.shape[-1]):
    #     plt.figure(figsize=(12, 5))
    #     plt.plot(np.c_[prediction.cpu().numpy()[:, ch], test_loader.dataset.target.cpu().numpy()[:, ch]])
    #     plt.legend([f'prediction channel{ch + 1}', f'label channel{ch + 1}'])
    #     if cfg['vis']:
    #         plt.show()

    return prediction, err


def main(signal_type, all_signal_type, cfg):
    """RNN_GC without NUE(non-uniform embedding) 算法的实现，对应论文中的算法2(返回格兰杰矩阵)
    1. 读取数据
    2. 分割 trial
        3. 分割数据集
        4. 求 WGCI 并保存
    """

    if signal_type in all_signal_type[:3]:
        origin_data = get_mat_data(f'Data/{signal_type}_noise1.mat', f'{signal_type}')    # 读取数据
    else:
        origin_data = get_mat_data(f'Data/{signal_type}.mat', f'{signal_type}')    # 读取数据

    # 存储 WGCI
    WGCI = []

    # 分割 trial
    for trial in range(cfg['trials']):
        start = int(trial * cfg['trial_points'] * cfg['overlap'])    # 50% 的 overlap
        idx = slice(start, cfg['trial_points'] + start)
        trial_set = origin_data[idx]

        # 训练集、验证集、测试集划分
        train_set, valid_set, test_set = time_series_split(trial_set, cfg['splits'])

        # 在完整数据集上训练模型
        model_id = 1
        print(f'model_id: {model_id}')

        _, err_all = train_net(train_set, valid_set, test_set, cfg['in_dim'], cfg['out_dim'], cfg)

        #  去掉某变量 训练网络
        temp = []
        for ch in range(cfg['num_channel']):
            model_id += 1
            print(f'model_id: {model_id}')
            idx = list(set(range(cfg['num_channel'])) - {ch})    # 剩余变量的索引
            # 把要去掉的数据的某维变为 0
            _, err = train_net(train_set[:, idx], valid_set[:, idx], test_set[:, idx], cfg['in_dim'] - 1, cfg['out_dim'] - 1, cfg)

            temp += [err]
        temp = torch.stack(temp)    # cfg['num_channel'] * num_point * out_dim

        # 扩充对角线
        err_cond = temp.new_zeros(temp.size(0), temp.size(1), cfg['num_channel'])
        for idx in range(cfg['num_channel']):
            col = list(set(range(cfg['num_channel'])) - {idx})
            err_cond[idx, :, col] = temp[idx]

        res = get_Granger_Causality(err_cond, err_all)
        # res = get_Granger_Causality1(err_cond, err_all)
        WGCI.append(res)
    return np.array(WGCI)


if __name__ == '__main__':
    # 基本设置
    timer = Timer()
    timer.start()
    config = get_json_data('configs/WGCI_cfg.json')
    device = set_device()
    all_signal_type = ['linear_signals', 'nonlinear_signals', 'longlag_nonlinear_signals', 'EEG64s']

    # RNN_GC
    # ground truth
    # ground_truth = np.zeros((5, 5))
    # ground_truth[0, 1] = 1
    # ground_truth[0, 2] = 1
    # ground_truth[0, 3] = 1
    # ground_truth[3, 4] = 1
    # ground_truth[4, 3] = 1
    # label = ['ch' + str(t + 1) for t in range(5)]
    # matshow(ground_truth, label, label, f'Ground truth Granger Causality Matrix', f'images/Ground_truth_Granger_Matrix.png')

    for signal_type in all_signal_type:
        gc_matrix = 0
        cfg = config[signal_type]
        for id in range(cfg['num_trials']):
            print(f'signal type: {signal_type} 实验 {id + 1}')
            gc_matrix += main(signal_type, all_signal_type, cfg)
        gc_matrix = np.squeeze(gc_matrix / cfg['num_trials'])
        gc_matrix[gc_matrix < cfg['threshold']] = 0.    # 阈值处理
        label = ['ch' + str(t + 1) for t in range(cfg['num_channel'])]
        matshow(gc_matrix, label, label, f'{signal_type} Granger Causality Matrix', f'images/without_NUE/{signal_type}_Granger_Matrix.png')

        # 保存结果
        np.savetxt(f'checkpoints/without_NUE/{signal_type}_granger_matrix.txt', gc_matrix)

    # 计时结束
    b = timer.stop()
