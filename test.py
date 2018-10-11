"""
Email: autuanliu@163.com
Date: 2018/10/10
主要用于测试代码功能是否完善
"""

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from core import MakeSeqData, Timer, get_mat_data
from models import Modeler, RNN_Net

# test Timer
tim = Timer()
tim.start()

# test MakeSeqData with DataLoader
data = get_mat_data('Data/test.mat')
print(data.shape)
dataset = MakeSeqData(data.transpose(), seq_length=20)
datasets = DataLoader(dataset, batch_size=32, shuffle=True)
for idx, (data, label) in enumerate(datasets):
    print(idx, data, label, data.shape, label.shape)

# test modeler

# get data and configures

# model configure
net = RNN_Net(5, 12, 1)
opt = optim.Adam(net.parameters(), lr=1e-3)
dataset = MakeSeqData(np.random.randn(2000, 5))
loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
configs = {
    'network': net,
    'opt': opt,
    'criterion': nn.MSELoss(),
    'dataloaders': {
        'train': loader,
        'valid': loader,
        'test': loader
    },
    'lrs_decay': optim.lr_scheduler.StepLR(opt, step_size=50),
    'epochs': 150,
    'hidden': net.initHidden(32)
}

# construct sub-model from BaseNet
sub_model = Modeler(**configs)
# train and test
sub_model.train()
sub_model.test()

# 计时结束
tim.stop()
