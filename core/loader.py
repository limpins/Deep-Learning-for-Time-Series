"""
Email: autuanliu@163.com
Date: 2018/9/28
"""

from inspect import isfunction

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from sklearn import preprocessing as skp
from torch.utils.data import DataLoader, Dataset


def get_mat_data(file_name, var_name):
    """从文件中读取出原始数据并转换为 numpy.array 类型

    Args:
        file_name (str): 数据存储的完整路径，如 'datastes/abc.mat'
        var_name (str): 存储数据的变量名

    Returns:
        numpy.array: 将读取到的原始数据转换为 numpy.ndarray 类型
    """

    data_dict = sio.loadmat(file_name)
    return data_dict[var_name]


def get_csv_data(file_name, sep=',', skiprows: int = 0, dtype=np.float32):
    data_df = pd.read_csv(file_name, sep=sep, skiprows=skiprows, dtype=dtype)
    return data_df.values


def get_txt_data(file_name, delimiter=',', dtype=np.float32):
    data = np.loadtxt(file_name, delimiter=delimiter, dtype=dtype)
    return data


def get_excel_data(file_name, sheet_name, skiprows=0, dtype=np.float32):
    data_df = pd.read_excel(file_name, sheet_name=sheet_name, skiprows=skiprows, dtype=dtype)
    return data_df.values


class MakeSeqData(Dataset):
    """创建序列数据集

    Args:
        data (numpy.ndarray): 序列数据
        seq_length (int): 序列的长度或者采样窗宽
        num_shift (int): 窗口每次平移的距离，默认为 1
        scaler (str): 数据标准化方式，默认为 'MinMaxScaler'，可选参数 ['MinMaxScaler', 'StandardScaler', None]
        或者自定义的 lambda 函数

    Returns:
        (torch.utils.data.Dataset) Dataset 子类，可以使用 DataLoader(数据类型为 tensor)
    """

    def __init__(self, data: np.ndarray, seq_length: int = 20, num_shift: int = 1, scaler_type: str = 'MinMaxScaler'):
        super(MakeSeqData, self).__init__()
        if scaler_type in ['MinMaxScaler', 'StandardScaler']:
            scaler = getattr(skp, scaler_type)()
            data = scaler.fit_transform(data)
        elif isfunction(scaler_type):
            data = scaler_type(data)
        else:
            raise ValueError("""An invalid option was supplied, options are ['MinMaxScaler', 'StandardScaler', None] or lambda function.""")
        # 获取子序列（窗口）数据
        num_point, _ = data.shape
        inputs, targets = [], []
        for idx in range(0, num_point - seq_length - num_shift + 1, num_shift):
            inputs.append(data[idx:(idx + seq_length), :])
            targets.append(data[idx + seq_length, :])
        self.data = torch.tensor(inputs)
        self.target = torch.tensor(targets)

    def get_tensor_data(self):
        """获取处理后的序列数据"""

        return self.data, self.target

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return self.data.shape[0]


def train_test_split(data: np.ndarray, split: float = 0.7):
    """序列数据的 train、test 划分

    Args:
        data (np.ndarray): 原始或者待划分的时间序列数据
        spilt (float, optional): Defaults to 0.8. 分割时序数据的分割比例，spilt之前的为训练集

    Returns:
        train_subseq, test_subseq (tuple): train_subseq, test_subseq
    """

    split_point = int(np.ceil(data.shape[0] * split))
    return data[:split_point, :], data[split_point:, :]


def make_loader(seq_data: np.ndarray, split: float = 0.7, seq_len: int = 20, bt_sz: int = 32):
    """获取可以迭代的分割后的数据。

    Args:
        seq_data (np.ndarray): 原始的序列数据
        split (float, optional): Defaults to 0.7. 训练集所占的比例
        seq_len (int, optional): Defaults to 20. 窗口的长度
        bt_sz (int, optional): Defaults to 32. batchsize

    Returns:
        [torch.utils.data.DataLoader]: train_loader, test_loader
    """

    # 数据分割
    train_subseq, test_subseq = train_test_split(seq_data, split=split)
    # 窗口采样
    sub = [train_subseq, test_subseq] = [MakeSeqData(t, seq_length=seq_len) for t in [train_subseq, test_subseq]]
    # 为了保证维度的匹配，需要去掉不满足一个batchsize的其余数据，测试集不需要随机打乱
    [train_loader, test_loader] = [DataLoader(t, batch_size=bt_sz, shuffle=sf, drop_last=True) for t, sf in zip(sub, [True, False])]
    return train_loader, test_loader
