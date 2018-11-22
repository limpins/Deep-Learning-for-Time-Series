"""
Email: autuanliu@163.com
Date: 2018/9/28

Ref:
1. Dynamic Granger causality based on Kalman filter for evaluation of functional network connectivity in fMRI data
"""

from inspect import isfunction

import numpy as np
import scipy.io as sio
from sklearn import preprocessing as skp


def get_mat_data(file_name, var_name):
    """从文件中读取出原始数据并转换为 np.array 类型

    Args:
        file_name (str): 数据存储的完整路径，如 'datastes/abc.mat'
        var_name (str): 存储数据的变量名

    Returns:
        np.array: 将读取到的原始数据转换为 np.ndarray 类型
    """

    data_dict = sio.loadmat(file_name)
    return data_dict[var_name]


def normalize(data, scaler_type: str = 'MinMaxScaler'):
    """标准化数据

    Args:
        data (np.ndarray): 未经过标准化的原始数据
        scaler_type (str, optional): Defaults to 'MinMaxScaler'. 归一化方式
    """

    if scaler_type in ['MinMaxScaler', 'StandardScaler']:
        data = getattr(skp, scaler_type)().fit_transform(data)
    elif isfunction(scaler_type):
        data = scaler_type(data)
    else:
        raise ValueError("""An invalid option was supplied, options are ['MinMaxScaler', 'StandardScaler', None] or lambda function.""")
    return data
