import numpy as np
import torch
from scipy.io import savemat
from torch import nn, optim

from Models import AdaBound, AdaBoundW, Modeler, RNN_Net
from core import (get_Granger_Causality, get_json_data, get_mat_data, make_loader, save_3Darray, set_device, time_series_split)


def get_person_WGCI(data_saved_name, data_field, cfg):
    """读取数据并求 WGCI

    1. 读取数据
    2. 分割 trial
        3. 分割数据集
        4. 求 WGCI 并保存

    Args:
        data_saved_name (str): 数据存储位置
        data_field (str): 存储的数据域名
        cfg (dict): 训练网络的配置
    """

    # 读取数据
    origin_data = get_mat_data(data_saved_name, data_field)

    # 存储 WGCI
    WGCI = []

    # 分割 trial
    for trial in range(cfg['trials']):
        start = int(trial * cfg['trial_points'] * cfg['overlap'])    # 50% 的 overlap
        idx = slice(start, cfg['trial_points'] + start)
        trial_set = origin_data[idx]

        # 训练集、验证集、测试集划分
        train_set, valid_set, test_set = time_series_split(trial_set, cfg['splits'])

        # 训练网络 获得 WGCI 数值
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
        WGCI.append(res)

    return np.array(WGCI)


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
    #     plt.legend([f'prediction EEG channel{ch + 1}', f'label EEG channel{ch + 1}'])
    #     if cfg['vis']:
    #         plt.show()

    return prediction, err


if __name__ == '__main__':
    # 设置参数
    cfg = get_json_data('configs/seizure.json')
    cfg0 = get_json_data('configs/Pp4Dp1.json')
    cfg1 = get_json_data('configs/pre_ictal.json')
    device = set_device()
    data_root = r'Data/'
    data_root1 = r'Data/pre_ictal.mat'
    save_root = r'seizure/'

    # Pp4->Dp1 pre_ictal 测试
    WGCI_Pp4Dp1_pre_ictal = 0
    for id in range(cfg0['num_trials']):
        print(f'Pp4->Dp1 pre ictal 实验 {id}')
        WGCI_Pp4Dp1_pre_ictal += get_person_WGCI(f'{data_root}Pp4Dp1_pre_ictal.mat', 'Pp4Dp1_pre_ictal', cfg0)
    WGCI_Pp4Dp1_pre_ictal /= cfg0['num_trials']
    # 阈值处理
    # WGCI_Pp4Dp1_pre_ictal[WGCI_Pp4Dp1_pre_ictal < cfg['threshold']] = 0.
    print(WGCI_Pp4Dp1_pre_ictal)
    save_3Darray(f'{save_root}WGCI_Pp4Dp1_pre_ictal.txt', WGCI_Pp4Dp1_pre_ictal)

    # Pp4->Dp1 ictal1 测试
    WGCI_Pp4Dp1_ictal1 = 0
    for id in range(cfg0['num_trials']):
        print(f'Pp4->Dp1 ictal1 实验 {id}')
        WGCI_Pp4Dp1_ictal1 += get_person_WGCI(f'{data_root}Pp4Dp1_ictal1.mat', 'Pp4Dp1_ictal1', cfg0)
    WGCI_Pp4Dp1_ictal1 /= cfg0['num_trials']
    # 阈值处理
    # WGCI_Pp4Dp1_ictal1[WGCI_Pp4Dp1_ictal1 < cfg['threshold']] = 0.
    print(WGCI_Pp4Dp1_ictal1)
    save_3Darray(f'{save_root}WGCI_Pp4Dp1_ictal1.txt', WGCI_Pp4Dp1_ictal1)

    # Pp4->Dp1 ictal2 测试
    WGCI_Pp4Dp1_ictal2 = 0
    for id in range(cfg0['num_trials']):
        print(f'Pp4->Dp1 ictal2 实验 {id}')
        WGCI_Pp4Dp1_ictal2 += get_person_WGCI(f'{data_root}Pp4Dp1_ictal2.mat', 'Pp4Dp1_ictal2', cfg0)
    WGCI_Pp4Dp1_ictal2 /= cfg0['num_trials']
    # 阈值处理
    # WGCI_Pp4Dp1_ictal2[WGCI_Pp4Dp1_ictal2 < cfg['threshold']] = 0.
    print(WGCI_Pp4Dp1_ictal2)
    save_3Darray(f'{save_root}WGCI_Pp4Dp1_ictal2.txt', WGCI_Pp4Dp1_ictal2)

    # seizure ictal1 计算
    WGCI_ictal1 = 0
    for id in range(cfg['num_trials']):
        print(f'seizure ictal1 实验 {id}')
        WGCI_ictal1 += get_person_WGCI(f'{data_root}ictal1.mat', 'ictal1', cfg)
    WGCI_ictal1 /= cfg['num_trials']
    # 阈值处理
    # WGCI_ictal1[WGCI_ictal1 < cfg['threshold']] = 0.
    print(WGCI_ictal1)
    save_3Darray(f'{save_root}WGCI_ictal1.txt', WGCI_ictal1)
    savemat(f'{save_root}WGCI_ictal1.mat', {'data': np.squeeze(WGCI_ictal1)})

    # seizure ictal2 计算
    WGCI_ictal2 = 0
    for id in range(cfg['num_trials']):
        print(f'seizure ictal2 实验 {id}')
        WGCI_ictal2 += get_person_WGCI(f'{data_root}ictal2.mat', 'ictal2', cfg)
    WGCI_ictal2 /= cfg['num_trials']
    # 阈值处理
    # WGCI_ictal2[WGCI_ictal2 < cfg['threshold']] = 0.
    print(WGCI_ictal2)
    save_3Darray(f'{save_root}WGCI_ictal2.txt', WGCI_ictal2)
    savemat(f'{save_root}WGCI_ictal2.mat', {'data': np.squeeze(WGCI_ictal2)})

    # pre_ictal
    WGCI_pre_ictal = 0
    for id in range(cfg1['num_trials']):
        print(f'pre_ictal 实验 {id}')
        WGCI_pre_ictal += get_person_WGCI(data_root1, 'pre_ictal', cfg1)

    WGCI_pre_ictal /= cfg1['num_trials']
    # 阈值处理
    # WGCI_pre_ictal[WGCI_pre_ictal < cfg['threshold']] = 0.
    print(WGCI_pre_ictal)
    save_3Darray(f'{save_root}WGCI_pre_ictal.txt', WGCI_pre_ictal)
    savemat(f'{save_root}WGCI_pre_ictal.mat', {'data': np.squeeze(WGCI_pre_ictal)})
