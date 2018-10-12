"""
Email: autuanliu@163.com
Date: 2018/10/11
"""

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from core import MakeSeqData, Timer, get_mat_data, set_device, train_test_split
from models import Modeler, RNN_Net

# 读取数据与数据预处理
bt_sz = 32
num_epoch = 5
seq_len = 20
device = set_device()
seqdata = get_mat_data('Data/linear_signals.mat', 'linear_signals')
train_subseq, test_subseq = train_test_split(seqdata, split=0.7)
train_subseq = MakeSeqData(train_subseq, seq_length=seq_len)
test_subseq = MakeSeqData(test_subseq, seq_length=seq_len)
# 为了保证维度的匹配，需要去掉不满足一个batchsize的其余数据
train_loader = DataLoader(train_subseq, batch_size=bt_sz, shuffle=True, drop_last=True)
test_loader = DataLoader(test_subseq, batch_size=bt_sz, drop_last=True)

# 完整数据集获取
x, y = MakeSeqData(seqdata, seq_length=20).get_tensor_data()


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
        test_loss = model.evaluate_model(test_loader)
        print(f'[{epoch+1}/{num_epoch}] ===>> train_loss: {train_loss: .4f} test_loss: {test_loss: .4f}')
    
    # 保存训练好的模型
    model.save_trained_model('checkpoints/model_weights.pth')

    # 预测\计算误差
    pred, err = model.predit_point_by_point(x, y)
    print(f'prediction error: {err:.4f}')
    timer.stop()

if __name__ == '__main__':
    train_valid()
