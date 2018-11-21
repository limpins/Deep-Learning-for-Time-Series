"""使用 Kalman Filter 实现对 MVAR(MultiVariate AutoRegression) 模型的系数估计。

1. https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html

Copyright:
----------
    Author: AutuanLiu
    Date: 2018/11/20
"""

import pathlib

import numpy as np
from filterpy.kalman import KalmanFilter

from utils import get_mat_data, normalize

# get data
file_path = './data/linear_signals.mat'
data = get_mat_data(file_path, 'linear_signals')

# 数据标准化
data = normalize(data)

# 构造 Kalman Filter
# 初始化
max_lag = 3
_, dim = data.shape
kf = KalmanFilter(max_lag*(dim**2), max_lag*dim)

# Assign the initial value for the state
# kf.x = np.zeros(kf.dim_x, 1)
# Define the state transition matrix
# kf.F = np.eye(kf.dim_x)
# Define the measurement function
tmp = data[:max_lag, :].T
kf.H = tmp[:, -1::-1]
# Define the covariance matrix(just multiply by the uncertainty)
# kf.P = np.eye(kf.dim_x)
# assign the measurement noise
# kf.R = np.eye(kf.dim_z)
# assign the process noise
# kf.Q = np.eye(kf.dim_x)

tmp1 = data[3:6, :].T
z = tmp1[:, -1::-1]
kf.predict()
kf.update(z)

# for z in data[max_lag:, :]:
#     kf.predict()
#     kf.update(z)
# print(kf.x)
