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
from torch.utils.data import Dataset, DataLoader
from .utils import train_test_split


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


def get_csv_data(file_name, sep=',', skiprows=0, dtype=np.float32):
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

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return self.data.shape[0]
