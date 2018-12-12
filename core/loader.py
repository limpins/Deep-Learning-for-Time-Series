"""
Email: autuanliu@163.com
Date: 2018/9/28
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def get_json_data(file_name):
    import ujson
    return ujson.load(open(file_name, 'r'))


def get_yaml_data(file_name):
    import yaml
    return yaml.load(open(file_name, 'r'))


def get_mat_data(file_name, var_name):
    """从文件中读取出原始数据并转换为 np.array 类型

    Args:
        file_name (str): 数据存储的完整路径，如 'datastes/abc.mat'
        var_name (str): 存储数据的变量名

    Returns:
        np.array: 将读取到的原始数据转换为 np.ndarray 类型
    """

    import scipy.io as sio
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


def train_test_split1(data: np.ndarray, split: float = 0.7):
    """序列数据的 train、test 划分

    Args:
        data (np.ndarray): 原始或者待划分的时间序列数据
        spilt (float, optional): Defaults to 0.8. 分割时序数据的分割比例，spilt之前的为训练集

    Returns:
        train_subseq, test_subseq (tuple): train_subseq, test_subseq
    """

    split_point = int(np.ceil(data.shape[0] * split))
    return data[:split_point, :], data[split_point:, :]


def train_test_split2(inputs: np.ndarray, targets: np.ndarray, split: float = 0.7):
    """序列窗口分割后数据的 train、test 划分

    Args:
        inputs (np.ndarray): 输入数据 x
        targets (np.ndarray): 输出数据 y
        spilt (float, optional): Defaults to 0.8. 分割数据的分割比例，spilt之前的为训练集

    Returns:
        X_train, X_test, y_train, y_test (np.ndarray)
    """

    split_point = int(np.ceil(inputs.shape[0] * split))
    return inputs[:split_point, :], inputs[split_point:, :], targets[:split_point, :], targets[split_point:, :]


def normalize(train_data, test_data, scaler_type: str = 'MinMaxScaler'):
    """归一化数据(一定是在训练集测试集已经划分完后进行)
    
    Args:
        train_data (np.ndarray): 未经过归一化的训练集原始数据
        test_data (np.ndarray): 未经过归一化的测试集原始数据
        scaler_type (str, optional): Defaults to 'MinMaxScaler'. 归一化方式
    """

    from inspect import isfunction
    from sklearn import preprocessing as skp

    if scaler_type in ['MinMaxScaler', 'StandardScaler']:
        scaler = getattr(skp, scaler_type)().fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
    elif isfunction(scaler_type):
        train_data = scaler_type(train_data)
        test_data = scaler_type(test_data)
    else:
        raise ValueError("""An invalid option was supplied, options are ['MinMaxScaler', 'StandardScaler', None] or lambda function.""")
    return train_data, test_data


def series2xy(series_data: np.ndarray, idx_x=None, idx_y=None, seq_length: int = 20, num_shift: int = 1):
    """将序列数据转换为监督学习数据
    
    Args:
        series_data (np.ndarray): 原始的序列数据
        idx_x (list or tuple or index slice or int): x 的索引位置, defaults to None.
        idx_y (list or tuple or index slice or int): y 的索引位置, defaults to None.
        seq_length (int, optional): Defaults to 20. 序列的长度或者采样窗宽
        num_shift (int, optional): Defaults to 1. 窗口每次平移的距离
    
    Returns:
        inputs, targets (np.ndarray)
    """

    # 获取子序列（窗口）数据
    num_point, _ = series_data.shape
    inputs, targets = [], []

    for idx in range(0, num_point - seq_length - num_shift + 1, num_shift):
        if idx_x is None and idx_y is None:
            inputs.append(series_data[idx:(idx + seq_length), :])
            targets.append(series_data[idx + seq_length, :])
        elif idx_x is None and idx_y is not None:
            inputs.append(series_data[idx:(idx + seq_length), :])
            targets.append(series_data[idx + seq_length, idx_y])
        elif idx_y is None and idx_x is not None:
            inputs.append(series_data[idx:(idx + seq_length), idx_x])
            targets.append(series_data[idx + seq_length, :])
        else:
            inputs.append(series_data[idx:(idx + seq_length), idx_x])
            targets.append(series_data[idx + seq_length, idx_y])
    return np.array(inputs), np.array(targets)


class MakeSeqData(Dataset):
    """创建序列数据集

    Args:
        inputs (np.ndarray): 输入数据 x
        targets (np.ndarray): 输出数据 y

    Returns:
        (torch.utils.data.Dataset) Dataset 子类，可以使用 DataLoader(数据类型为 tensor)
    """

    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        super(MakeSeqData, self).__init__()
        # 这里要保证维度是不可改变的 [2, 1] 和 [2,] 是不同的维度
        self.fill_dim = lambda a: a.unsqueeze_(1) if a.ndimension() == 1 else a
        self.data = self.fill_dim(torch.from_numpy(inputs))
        self.target = self.fill_dim(torch.from_numpy(targets))

    def get_tensor_data(self):
        """获取处理后的序列数据"""

        return self.data, self.target

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return self.data.shape[0]


def make_loader(seq_data: np.ndarray, idx_x=None, idx_y=None, tt_split=0.7, tv_split=0.7, seq_len=20, num_shift=1, bt_sz=32, s_type='MinMaxScaler'):
    """获取可以迭代的分割后的数据

    Args:
        seq_data (np.ndarray): 原始的序列数据
        idx_x (list or tuple or index slice or int): x 的索引位置, defaults to None.
        idx_y (list or tuple or index slice or int): y 的索引位置, defaults to None.
        tt_split (float, optional): Defaults to 0.7. 训练集测试集划分时训练集所占的比例
        tv_split (float, optional): Defaults to 0.7. 训练集验证集划分时训练集所占的比例
        seq_len (int, optional): Defaults to 20. 窗口的长度
        num_shift (int, optional): Defaults to 1. 窗口每次平移的距离
        bt_sz (int, optional): Defaults to 32. batchsize
        s_type (str, optional): Defaults to 'MinMaxScaler'. 归一化方式

    Returns:
        [torch.utils.data.DataLoader]: train_loader, valid_loader, test_loader
    """

    # 训练集、测试集数据分割
    train_subseq, test_subseq = train_test_split1(seq_data, split=tt_split)
    # 全部数据归一化
    train_subseq, test_subseq = normalize(train_subseq, test_subseq, scaler_type=s_type)
    # 转为窗口数据
    X_train, y_train = series2xy(train_subseq, idx_x=idx_x, idx_y=idx_y, seq_length=seq_len, num_shift=num_shift)
    X_test, y_test = series2xy(test_subseq, idx_x=idx_x, idx_y=idx_y, seq_length=seq_len, num_shift=num_shift)
    # 训练集、验证集数据分割(已经归一化)
    X_train, X_valid, y_train, y_valid = train_test_split2(X_train, y_train, split=tv_split)
    # 构造数据集
    sub = [train_subseq, valid_subseq, test_subseq] = [MakeSeqData(x, y) for x, y in zip([X_train, X_valid, X_test], [y_train, y_valid, y_test])]
    # 为了保证维度的匹配，需要去掉不满足一个batchsize的其余数据，测试集不需要随机打乱
    [train_loader, valid_loader, test_loader] = [DataLoader(t, batch_size=bt_sz, shuffle=sf, drop_last=False) for t, sf in zip(sub, [True, False, False])]
    return train_loader, valid_loader, test_loader
