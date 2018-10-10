"""
Email: autuanliu@163.com
Date: 2018/10/10
主要用于测试代码功能是否完善
"""

from core import get_mat_data, MakeSeqData
from torch.utils.data import DataLoader

data = get_mat_data('Data/test.mat')
print(data.shape)
dataset = MakeSeqData(data.transpose(), 20)
datasets = DataLoader(dataset, batch_size=32, shuffle=True)
for idx, (data, label) in enumerate(datasets):
    print(idx, data.shape, label.shape)
    if idx == 10:
        print(data, label)
