"""使用 Extend Kalman Filter 实现对 MVARX 模型的系数估计。

Copyright:
----------
    Author: AutuanLiu
    Date: 2018/11/20
"""

import numpy as np

from core import *

# !非线性模型
terms_path = './kalman_filter/data/nonlinear_terms.mat'
term = Selector(terms_path)
terms_repr = term.make_terms()

# *保存候选项集合
fname = './kalman_filter/data/nonlinear_candidate_terms.txt'
np.savetxt(fname, terms_repr, fmt='%s')

# *selection
Kalman_H, candidate_terms, terms_No, max_lag = term.make_selection()

# *非线性数据
file_path = './kalman_filter/data/nonlinear_signals5D_noise1.mat'
data = get_mat_data(file_path, 'nonlinear_signals')

# 数据标准化
data = normalize(data)

# *构造 Kalman Filter
kf = Kalman4FROLS(data, Kalman_H=Kalman_H, uc=0.01)
y_coef = kf.estimate_coef()
print(y_coef)

# *估计模型生成
est_model = make_func4K4FROLS(y_coef, candidate_terms, terms_No, fname='./kalman_filter/data/K4FROLS_est_model.txt')
print(est_model)