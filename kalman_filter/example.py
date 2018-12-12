"""测试 Kalman Filter 估计模型系数。

Copyright:
----------
    Author: AutuanLiu
    Date: 2018/11/20
"""

import numpy as np

from filter import Kalman4ARX, Kalman4FROLS
from tools import *
from utils import *
from selector import Selector

# !Kalman4ARX 测试
# timer = Timer()
# timer.start()

# get data 线性模型
file_path = './kalman_filter/data/linear_signals5D_noise1.mat'
data = get_mat_data(file_path, 'linear_signals')

# 数据标准化
data = normalize(data)

# *构造 Kalman Filter
kf = Kalman4ARX(data, 5, uc=0.01)
y_coef, A_coef = kf.estimate_coef(0.1)
print(y_coef, A_coef)

# *估计模型生成
est_model = make_linear_func(A_coef, var_name='x', fname='./kalman_filter/data/linear_est_model.txt')
print(est_model)

# !Kalman4FROLS 测试
# !Selector 测试
# !线性模型
terms_path = './kalman_filter/data/linear_terms.mat'
term = Selector(terms_path)
terms_repr = term.make_terms()

# *保存候选项集合
fname = './kalman_filter/data/linear_candidate_terms.txt'
np.savetxt(fname, terms_repr, fmt='%s')

# *selection
Kalman_H, candidate_terms, terms_No, max_lag = term.make_selection()

# *非线性数据
file_path = './kalman_filter/data/linear_signals5D_noise1.mat'
data = get_mat_data(file_path, 'linear_signals')

# 数据标准化
data = normalize(data)

# *构造 Kalman Filter
kf = Kalman4FROLS(data, Kalman_H=Kalman_H, max_lag=max_lag, uc=0.01)
y_coef = kf.estimate_coef()
print(y_coef)

# *估计模型生成
est_model = make_func4K4FROLS(y_coef, candidate_terms, terms_No, fname='./kalman_filter/data/K4FROLS_est_model.txt')
print(est_model)

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
kf = Kalman4FROLS(data, Kalman_H=Kalman_H, max_lag=max_lag, uc=0.01)
y_coef = kf.estimate_coef()
print(y_coef)

# *估计模型生成
est_model = make_func4K4FROLS(y_coef, candidate_terms, terms_No, fname='./kalman_filter/data/K4FROLS_est_model.txt')
print(est_model)
