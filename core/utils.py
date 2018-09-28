"""
Email: autuanliu@163.com
Date: 2018/9/28
"""

import datetime as dt

import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing


class Data_pre:
    """Data pre-processing.
    """

    def get_batch_sequence(self, x, seq_length=10, num_shift=1):
        """Get batch sequence data.

        Args:
            x (matrix): num_points * num_channel.
            seq_length (int, optional): Defaults to 10. length of sequence.
            num_shift (int, optional): Defaults to 1. step of slide windows.

        Returns:
            tuple: (samples, timesteps, input_dim)
        """

        num_points = x.shape[0]
        inputs = []
        targets = []
        # for p in np.arange(0, num_points, max(num_shift, seq_length // 5)):
        for p in np.arange(0, num_points, num_shift):
            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            if p + seq_length + num_shift >= num_points:
                break

            inputs.append(x[p:(p + seq_length), :])
            # targets.append(x[p + num_shift: p + seq_length + num_shift, :])
            targets.append(x[p + seq_length, :])
        inputs = np.array(inputs)
        targets = np.array(targets)
        # 随机取数据
        idx = np.random.permutation(np.arange(inputs.shape[0]))
        inputs = inputs[idx]
        targets = targets[idx]

        return inputs, targets

    def get_seq_data(self, file_name, scaler='min-max'):
        """Get sequence data.

        Args:
            file_name (str): file name
            scaler (str, optional): Defaults to 'min-max'. method for normalizing data.('min-max', 'standard')
        """

        data = sio.loadmat(file_name)['data'].transpose()  # load '*.mat' data
        self.num_channel = data.shape[1]
        # scaler = preprocessing.StandardScaler().fit(data) # Data normalization
        # data = scaler.transform(data)
        min_max_scaler = preprocessing.MinMaxScaler()
        data = min_max_scaler.fit_transform(data)  # scale data to [0. 1]
        x, y = self.get_batch_sequence(
            data, num_shift=self.num_shift, seq_length=self.seq_length)


class Timer():

    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        print(f'Time taken: {end_dt - self.start_dt}')


def set_device(): return torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
