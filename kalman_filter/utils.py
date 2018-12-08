"""
Email: autuanliu@163.com
Date: 2018/11/21

Ref:
1. Dynamic Granger causality based on Kalman filter for evaluation of functional network connectivity in fMRI data
2. https://stackoverflow.com/questions/3685265/how-to-write-a-multidimensional-array-to-a-text-file
"""

import datetime as dt

import numpy as np

__all__ = ['get_mat_data', 'normalize', 'save_2Darray', 'save_3Darray', 'Timer', 'make_linear_func']


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


def normalize(data, scaler_type: str = 'MinMaxScaler'):
    """标准化数据

    Args:
        data (np.ndarray): 未经过标准化的原始数据
        scaler_type (str, optional): Defaults to 'MinMaxScaler'. 归一化方式
    """

    from inspect import isfunction
    from sklearn import preprocessing as skp
    if scaler_type in ['MinMaxScaler', 'StandardScaler']:
        data = getattr(skp, scaler_type)().fit_transform(data)
    elif isfunction(scaler_type):
        data = scaler_type(data)
    else:
        raise ValueError("""An invalid option was supplied, options are ['MinMaxScaler', 'StandardScaler', None] or lambda function.""")
    return data


def save_2Darray(file_path, data):
    """save np.array(2D) into txt file.(Ref2)

    Args:
        file_path (str or instance of Path(windowns or linux)): the file path to save data.
        data (np.array): the data need be saved.
    """

    with open(file_path, 'w') as outfile:
        outfile.write(f'# Array shape: {data.shape}\n')
        np.savetxt(outfile, data, fmt='%.4f')


def save_3Darray(file_path, data):
    """save np.array(3D) into txt file.(Ref2)

    Args:
        file_path (str or instance of Path(windowns or linux)): the file path to save data.
        data (np.array): the data need be saved.
    """

    with open(file_path, 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write(f'# Array shape: {data.shape}\n')

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in data:
            np.savetxt(outfile, data_slice, fmt='%.4f')
            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')


class Timer():
    """计时器类"""

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        print(f'Time taken: {(end_dt - self.start_dt).total_seconds():.2f}s')


def make_linear_func(A_coef, var_name: str = 'y', step_name: str = 't', save=True, fname='est_model.txt', **kwargs):
    """根据系数生成模型表达式
    
    Args:
        A_coef (np.array): y = Ax + e 形式的模型系数矩阵 A
        var_name (str, optional): Defaults to 'y'. 使用的变量名
        step_name (str, optional): Defaults to 't'. 时间点变量名
        save (bool, optional): Defaults to True. 是否保存结果
        kwargs: np.savetxt args
    
    Returns:
        func_repr (np.array) 模型表达式
    """

    max_lag, n_dim, _ = A_coef.shape
    func_repr = []
    for var in range(n_dim):
        y = f'{var_name}{var+1}({step_name}) = '
        for dim in range(n_dim):
            for lag in range(max_lag):
                if abs(A_coef[lag, var, dim]) > 0.:
                    y += f'{A_coef[lag, var, dim]:.4f} * {var_name}{dim+1}({step_name}-{lag+1}) + '
        func_repr.append(y + f'e{var+1}({step_name})')
    func_repr = np.array(func_repr).reshape(-1, 1)
    if save:
        np.savetxt(fname, func_repr, fmt='%s', **kwargs)
    return func_repr
