"""测试 Kalman Filter 估计模型系数。

Copyright:
----------
    Author: AutuanLiu
    Date: 2018/11/20
"""

import numpy as np

from filter import Linear_Kalman_Estimation
from tools import *
from utils import *
from selector import Selector

## Kalman Filter 测试
# timer = Timer()
# timer.start()

# get data
file_path = './kalman_filter/data/linear_signals5D_noise1.mat'
data = get_mat_data(file_path, 'linear_signals')

# 数据标准化
data = normalize(data)

# # 构造 Kalman Filter
# kf = Linear_Kalman_Estimation(data, 5, uc=0.01)
# y_coef, A_coef = kf.estimate_coef(0.1)


## Selector 测试
term = Selector(data, 2, 5, 0.0001)
term.max_lag = 2
term.ndim = 3
print(term.buildV(3))
