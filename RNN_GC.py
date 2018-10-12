"""
Email: autuanliu@163.com
Date: 2018/9/28
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from core import MakeSeqData, Timer, get_mat_data, set_device, train_test_split
from models import Modeler, RNN_Net

# 读取数据与数据预处理
bt_sz = 32
num_epoch = 50
seq_len = 20
device = set_device()
seqdata = get_mat_data('Data/linear_signals.mat', 'linear_signals')
train_subseq, valid_subseq = train_test_split(seqdata, split=0.8)
train_subseq = MakeSeqData(train_subseq, seq_length=seq_len)
valid_subseq = MakeSeqData(valid_subseq, seq_length=seq_len)
train_loader = DataLoader(train_subseq, batch_size=bt_sz,
                          shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_subseq, batch_size=bt_sz, drop_last=True)


# 构建模型与配置
def train_valid():
    timer = Timer()
    timer.start()
    model = RNN_Net(5, 15, 5, rnn_type='LSTM', num_layers=1)
    opt = optim.RMSprop(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.MSELoss()
    model = Modeler(model, opt, criterion, device, batchsize=bt_sz)

    for epoch in range(num_epoch):
        train_loss = model.train_model(train_loader)
        valid_loss = model.evaluate_model(valid_loader)
        print(f'[{epoch+1}/{num_epoch}] ===>> train_loss: {train_loss: .4f} valid_loss: {valid_loss: .4f}')
    
    # 保存训练好的模型
    model.save_trained_model('checkpoints/model_weights.pth')
    timer.stop()

if __name__ == '__main__':
    train_valid()
