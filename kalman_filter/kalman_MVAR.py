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
from tools import *

# get data
file_path = './kalman_filter/data/linear_signals5D_noise1.mat'
data = get_mat_data(file_path, 'linear_signals')

# 数据标准化
data = normalize(data)

# 网格搜索
lag_range = range(1, 20)
uc_range = np.arange(0.001, 0.01, 0.0005)
# best_lag_uc = grid_search1(data, lag_range, uc_range)
# print(f'best_lag_uc: {best_lag_uc}')
best_lag, _ = grid_search2(data, lag_range, 0.001, plot=True)
print(f'best_lag: {best_lag}')
best_uc, _ = grid_search3(data, 5, uc_range, plot=True)
print(f'best_uc: {best_uc}')

# 构造 Kalman Filter
# 初始化
# max_lag = 19
# kf = Linear_Kalman_Estimation(data, max_lag, uc=0.0095)
# y_coef, A_coef = kf.estimate_coef(0.1)

# # 保存结果
# np.savetxt('./kalman_filter/data/y_coef.txt', y_coef, fmt='%.4f', delimiter=', ')