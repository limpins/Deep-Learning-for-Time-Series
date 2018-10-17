"""
Email: autuanliu@163.com
Date: 2018/10/10
主要用于测试代码功能是否完善
"""

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from core import MakeSeqData, Timer, get_mat_data, set_device, make_loader
from models import Modeler, RNN_Net


# test Timer
tim = Timer()
tim.start()

# test MakeSeqData with DataLoader
data = get_mat_data('Data/test.mat', 'data')
print(data.shape)
dataset = MakeSeqData(data.transpose(), seq_length=20)
datasets = DataLoader(dataset, batch_size=32, shuffle=True)
for idx, (data, label) in enumerate(datasets):
    pass

# test modeler

# get data and configures

net = RNN_Net(5, 12, 5)
opt = optim.Adam(net.parameters(), lr=1e-3)
seqdata_all = torch.randn(2000, 5)
train_loader, test_loader = make_loader(seqdata_all, split=0.7, seq_len=20, bt_sz=32)
criterion = nn.MSELoss(reduction='elementwise_mean')
device = set_device()

# construct sub-model from BaseNet
sub_model = Modeler(net, opt, criterion, device, batchsize=32)
# train and test
for epoch in range(10):
    loss1 = sub_model.train_model(train_loader)
    loss2 = sub_model.evaluate_model(test_loader)

# 计时结束
tim.stop()
