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

from core import MakeSeqData, Timer, get_mat_data, set_device, train_test_split
from models import Modeler, RNN_Net


def make_loader(seq_data, split=0.7, seq_len=20, bt_sz=32):
    train_subseq, test_subseq = train_test_split(seq_data, split=0.7)
    train_subseq = MakeSeqData(train_subseq, seq_length=seq_len)
    test_subseq = MakeSeqData(test_subseq, seq_length=seq_len)
    # 为了保证维度的匹配，需要去掉不满足一个batchsize的其余数据
    train_loader = DataLoader(train_subseq, batch_size=bt_sz, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_subseq, batch_size=bt_sz, drop_last=True)
    return train_loader, test_loader


# 构建模型与配置
def train_valid(in_dim, hidden_dim, out_dim, ckpt, x, y, train_loader, test_loader):
    model = RNN_Net(in_dim, hidden_dim, out_dim, rnn_type='LSTM', num_layers=2)
    opt = optim.RMSprop(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.MSELoss()
    model = Modeler(model, opt, criterion, device, batchsize=bt_sz)

    for epoch in range(num_epoch):
        train_loss = model.train_model(train_loader)
        test_loss = model.evaluate_model(test_loader)
        print(f'[{epoch+1}/{num_epoch}] ===>> train_loss: {train_loss: .4f} test_loss: {test_loss: .4f}')
    
    # 保存训练好的模型
    # model.save_trained_model('checkpoints/model_weights.pth')
    model.save_trained_model(ckpt)

    # 预测\计算误差
    err = model.predit_point_by_point(x, y)[1]
    return err

def main():
    """RNN_GC 算法的实现，对应论文中的算法1(返回格兰杰矩阵)
    """

    # 读取数据
    seqdata = get_mat_data(f'Data/{signal_type}.mat', f'{signal_type}')
    # 完整数据集训练模型
    train_loader, test_loader = make_loader(seqdata, split=0.7, seq_len=20, bt_sz=32)
    x, y = MakeSeqData(seqdata, seq_length=20).get_tensor_data()
    err = train_valid(5, 15, 5, f'checkpoints/{signal_type}_model_weights.pth', x, y, train_loader, test_loader)
    # 去掉一个变量训练模型
    deno = []
    for ch in range(num_channel):
        idx = list(set(range(num_channel)) - {ch})
        seq_data = seqdata[:, idx]
        train_loader, test_loader = make_loader(seq_data, split=0.7, seq_len=20, bt_sz=32)
        x, y = MakeSeqData(seq_data, seq_length=20).get_tensor_data()
        err1 = train_valid(4, 15, 4, f'checkpoints/{signal_type}_model_weights{ch}.pth', x, y, train_loader, test_loader)
        deno += [err1.var(0)]
    frac = torch.stack(deno)
    frac1 = frac.new_zeros(num_channel, num_channel)
    for idx in range(num_channel):
        col = list(set(range(num_channel)) - {idx})
        frac1[idx, col] = frac[idx, :]
    
    gc_matrix = frac1 / err.var(0)
    for i in range(num_channel):
        gc_matrix[i, i] = gc_matrix.new_ones(1)
    gc_matrix[gc_matrix < 1] = 1  # 格兰杰因果>0

    return gc_matrix.log().cpu().numpy()



if __name__ == '__main__':
    # 基本设置
    timer = Timer()
    timer.start()
    bt_sz = 32
    num_epoch = 50
    num_channel = 5
    seq_len = 20
    num_trial = 5
    device = set_device()
    all_signal_type = ['linear_signals', 'nonlinear_signals', 'longlag_nonlinear_signals']
    signal_type = 'linear_signals'

    # RNN_GC
    avg_gc_matrix = 0
    for signal_type in all_signal_type:
        print(f'signal type: {signal_type}')
        for tr in range(num_trial):
            avg_gc_matrix += main()
        avg_gc_matrix /= 5.
        avg_gc_matrix[avg_gc_matrix<0.01] = 0  # 阈值处理
        plt.matshow(avg_gc_matrix)
        np.savetxt(f'checkpoints/{signal_type}_granger_matrix.txt', avg_gc_matrix)
        plt.savefig(f'images/{signal_type}.png')

    # 计时结束
    timer.stop()
