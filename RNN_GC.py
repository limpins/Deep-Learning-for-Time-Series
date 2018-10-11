"""
Email: autuanliu@163.com
Date: 2018/9/28
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from core import MakeSeqData, Timer, get_mat_data, train_test_split, set_device, repackage_hidden
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
valid_loader = DataLoader(valid_subseq, batch_size=bt_sz)

# 构建模型与配置
model = RNN_Net(5, 15, 5, rnn_type='LSTM', num_layers=2)
print(next(model.parameters()))
model = nn.DataParallel(model).to(
    device) if torch.cuda.device_count() > 1 else model.to(device)
hidden = model.initHidden(bt_sz)  # 获取RNN模型的初始状态
opt = optim.RMSprop(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.MSELoss()
# 训练模型
model.train()
for epoch in range(num_epoch):
    for data, target in train_loader:
        data, target = data.to(device).float(), target.to(device).float()
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        #
        # (data.type())
        out, hidden = model(data, hidden)
        # hidden = (hidden[0].data, hidden[1].data)
        loss = criterion(out, target)

        # 后向传播
        opt.zero_grad()
        loss.backward()
        opt.step()

    # 每个 epoch 输出训练结果
    print(f'Epoch [{epoch+1}/{num_epoch}], Loss [{loss.item():.4f}]')
