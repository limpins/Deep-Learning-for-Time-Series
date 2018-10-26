"""
Email: autuanliu@163.com
Date: 2018/10/10
主要用于测试代码功能是否完善
"""

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from core import (MakeSeqData, Timer, get_json_data, get_mat_data,
                  get_yaml_data, make_loader, series2xy, set_device,
                  train_test_split1, train_test_split2, normalize)
from models import Modeler, RNN_Net

# test Timer
tim = Timer()
tim.start()

# test MakeSeqData with DataLoader
data = get_mat_data('Data/test.mat', 'data')
print(data.shape)
X, y = series2xy(data.transpose())
dataset = MakeSeqData(X, y)
datasets = DataLoader(dataset, batch_size=32, shuffle=True)
for idx, (data, label) in enumerate(datasets):
    pass

# test modeler

# get data and configures
net = RNN_Net(5, 12, 5)
opt = optim.Adam(net.parameters(), lr=1e-3)
lr_decay = optim.lr_scheduler.ReduceLROnPlateau(opt)
seq = get_mat_data(f'Data/linear_signals.mat', f'linear_signals')
# train_test_split1 测试
train_sub, test_sub = train_test_split1(seq)
print(train_sub, test_sub, train_sub.shape, test_sub.shape)
# 归一化测试
train_sub, test_sub = normalize(train_sub, test_sub)
print(train_sub, test_sub, train_sub.shape, test_sub.shape)

# series2xy 测试
X_train, y_train = series2xy(train_sub)
print(X_train, y_train)

# train_test_split2 测试
X_train, X_valid, y_train, y_valid = train_test_split2(X_train, y_train, split=0.7)
print(X_train, X_valid, y_train, y_valid)
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

train_loader, valid_loader, test_loader = make_loader(seq, tt_split=0.7, tv_split=0.7, seq_len=20, bt_sz=16)
print('test_loader.dataset', test_loader.dataset.data)
print('test_loader.dataset', test_loader.dataset.get_tensor_data())
criterion = nn.MSELoss(reduction='elementwise_mean')
device = set_device()

print(next(iter(train_loader))[0].size(0))

# construct sub-model from BaseNet
sub_model = Modeler(net, opt, criterion, device, visualization=True)
# train and test
for epoch in range(1):
    loss1 = sub_model.train_model(train_loader, epoch)
    loss2 = sub_model.evaluate_model(test_loader, epoch)
    lr_decay.step(loss2)

# 计时结束
tim.stop()

x = get_json_data('configs/cfg.json')
y = get_yaml_data('configs/cfg.yaml')
print(x)
print(y)
