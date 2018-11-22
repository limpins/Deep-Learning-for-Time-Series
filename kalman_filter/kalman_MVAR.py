"""使用 Kalman Filter 实现对 MVAR(MultiVariate AutoRegression) 模型的系数估计。

1. https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html

Copyright:
----------
    Author: AutuanLiu
    Date: 2018/11/20
"""

import pathlib

import numpy as np
from filter import Linear_Kalman_Estimation

from utils import get_mat_data, normalize

# get data
file_path = './data/linear_signals.mat'
data = get_mat_data(file_path, 'linear_signals')

# 数据标准化
data = normalize(data)

# 构造 Kalman Filter
# 初始化
max_lag = 5
kf = Linear_Kalman_Estimation(data, max_lag, uc=0.01)

y_coef, A_coef = kf.estimate_coef(0.3)
print(y_coef)
print(A_coef)
print(A_coef.shape)
# print(kf.forward()[0])

# 保存结果
np.savetxt('./kalman_filter/data/y_coef.txt', y_coef, fmt='%.4f', delimiter=', ')
