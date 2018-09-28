"""
Email: autuanliu@163.com
Date: 2018/9/28
"""

import numpy as np
import scipy.io as sio


class Data_pre:
    """Data pre-processing.
    """
    @staticmethod
    def get_batch_sequence(x, seq_length=10, num_shift=1):
        """Get sequence data.

        Args:
            x (matrix): num_points * num_channel.
            seq_length (int, optional): Defaults to 10. length of sequence.
            num_shift (int, optional): Defaults to 1. [description]

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
