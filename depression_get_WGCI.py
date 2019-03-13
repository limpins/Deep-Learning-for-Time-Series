"""
Email: autuanliu@163.com
Date: 2018/10/11
"""

import numpy as np
import torch
from torch import nn, optim

from core import (Timer, get_Granger_Causality, get_json_data, get_mat_data, make_loader, matshow, set_device, save_3Darray)
from Models import Modeler, RNN_Net
from tools import cyclical_lr


def train_valid(in_dim, hidden_dim, out_dim, ckpt, test_data, loaders):
    """训练与验证模型，每个epoch都进行训练与验证
    """

    net = RNN_Net(in_dim, hidden_dim, out_dim, rnn_type=cfg['rnn_type'], num_layers=cfg['num_layers'], dropout=cfg['dropout'])    # 创建模型实例
    opt = optim.RMSprop(net.parameters(), lr=cfg['lr_rate'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])    # 优化器定义
    lr_decay2 = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)    # 学习率衰减
    # CLR policy
    # step_size = 5
    # clr = cyclical_lr(step_size, min_lr=0.001, max_lr=0.005)
    # lr_decay1 = lr_scheduler.LambdaLR(optimizer, [clr])
    criterion = nn.MSELoss()    # 损失函数定义，由于是回归预测，所以设为 MSE loss
    model = Modeler(net, opt, criterion, device)

    val_loss = []
    min_val_loss = 0.5
    for epoch in range(cfg['num_epoch']):
        # lr_decay1.step()
        train_loss = model.train_model(loaders['train'])    # 当前 epoch 的训练损失
        valid_loss = model.evaluate_model(loaders['valid'])    # 当前 epoch 的验证损失
        lr_decay2.step(valid_loss)

        # 增加 early_stopping 策略
        # if valid_loss <= min_val_loss:
        #     min_val_loss = valid_loss
        #     model.save_trained_model(ckpt)
        print(f"[{epoch+1}/{cfg['num_epoch']}] ===>> train_loss: {train_loss: .4f} | valid_loss: {valid_loss: .4f}")

    # 预测并计算误差
    model.save_trained_model(ckpt)
    model.load_best_model(ckpt)    # 使用最好的模型进行预测
    # model.save_model('model.pth')
    err = model.predit_point_by_point(*test_data)[1]
    return err


def main(seqdata_all):
    """RNN_GC without NUE(non-uniform embedding) 算法的实现，对应论文中的算法2(返回格兰杰矩阵)
    """
    # 在完整数据集上训练模型
    model_id = 1
    print(f'model_id: {model_id}')
    train_loader, valid_loader, test_loader = make_loader(
        seqdata_all, tt_split=cfg['tt_split'], tv_split=cfg['tv_split'], seq_len=cfg['seq_len'], bt_sz=cfg['bt_sz'])
    loaders = {'train': train_loader, 'valid': valid_loader}
    err_all = train_valid(cfg['in_dim'], cfg['num_hidden'], cfg['out_dim'], f'checkpoints/depression/model_weights.pth',
                          test_loader.dataset.get_tensor_data(), loaders)

    # 去掉一个变量训练模型
    temp = []
    for ch in range(cfg['num_channel']):
        model_id += 1
        print(f'model_id: {model_id}')
        idx = list(set(range(cfg['num_channel'])) - {ch})    # 剩余变量的索引
        seq_data = seqdata_all[:, idx]    # 当前的序列数据
        train_loader, valid_loader, test_loader = make_loader(
            seq_data, tt_split=cfg['tt_split'], tv_split=cfg['tv_split'], seq_len=cfg['seq_len'], bt_sz=cfg['bt_sz'])
        loaders = {'train': train_loader, 'valid': valid_loader}
        err = train_valid(cfg['in_dim'] - 1, cfg['num_hidden'], cfg['out_dim'] - 1, f'checkpoints/depression/model_weights{ch}.pth',
                          test_loader.dataset.get_tensor_data(), loaders)
        temp += [err]
    temp = torch.stack(temp)    # cfg['num_channel'] * num_point * out_dim

    # 扩充对角线
    err_cond = temp.new_zeros(temp.size(0), temp.size(1), cfg['num_channel'])
    for idx in range(cfg['num_channel']):
        col = list(set(range(cfg['num_channel'])) - {idx})
        err_cond[idx, :, col] = temp[idx]
    return get_Granger_Causality(err_cond, err_all)


if __name__ == '__main__':
    # 基本设置
    cfg= get_json_data('configs/depression.json')
    device = set_device()

    # 对每个患者读取数据
    for patient in range(1, 70):
        data = get_mat_data(f'Data/depression/{patient}.mat', 'MK')    # 读取数据
        data = data[:, 1:]
        WGCI = []
        for trial in range(cfg['trial']):
            print(f'patient ID: {patient}, trial: {trial}')
            timer = Timer()
            timer.start()
            tmp = trial * cfg['n_point'] // 2
            idx = slice(tmp, cfg['n_point'] + tmp)
            gc_matrix = main(data[idx])  # 求WGCI
            # print(gc_matrix)
            WGCI.append(gc_matrix)
            # 计时结束
            b = timer.stop()
        WGCI = np.array(WGCI)
        save_3Darray(f'depression/WGCI_{patient}.txt', WGCI)
    # label = ['ch' + str(t + 1) for t in range(cfg['num_channel'])]
    # matshow(avg_gc_matrix, label, label, f'{signal_type} Granger Causality Matrix', f'images/without_NUE/{signal_type}_Granger_Matrix.png')
