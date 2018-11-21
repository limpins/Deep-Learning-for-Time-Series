"""重载 filterpy.kalman 模块中的 KalmanFilter，以满足估计系数的需要，因为在估计系数时需要同时更新 Q 与 R。

1. https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html

文献：
1.	Accurate epileptogenic focus localization through time-variant functional connectivity analysis of intracranial electroencephalographic signals
2.	A new Kalman filter approach for the estimation of high-dimensional time-variant multivariate AR models and its application in analysis of laser-evoked brain potentials
3.	Dynamic Granger causality based on Kalman filter for evaluation of functional network connectivity in fMRI data
4.	Seizure-Onset Mapping Based on Time-Variant Multivariate Functional Connectivity Analysis of High-Dimensional Intracranial EEG: A Kalman Filter Approach

Copyright:
----------
    Author: AutuanLiu
    Date: 2018/11/21
"""

from copy import deepcopy

import numpy as np
from filterpy.common import pretty_str, reshape_z
from filterpy.kalman import KalmanFilter
from numpy import dot, eye, isscalar, shape, zeros


class Linear_Kalman_filter(KalmanFilter):
    """定义适用于估计时间序列模型系数的kalman filter。"""

    def __init__(self, dim_x, dim_z, dim_u=0):
        super().__init__(dim_x, dim_z, dim_u=dim_u)
        self.init()

    def init(self):
        pass

    def update(self, z, lambda=1e-4, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter.

        If z is None, nothing is computed. However, x_post and P_post are
        updated with the prior (x_prior, P_prior), and self.z is set to None.

        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.

        lambda: float, scalar
            update coefficient, forgetting factor.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        H : np.array, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z = np.array([[None]*self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = np.zeros((self.dim_z, 1))
            return

        z = reshape_z(z, self.dim_z, self.x.ndim)

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        if H is None:
            H = self.H

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(H, self.x)

        # common subexpression for speed
        PHT = dot(self.P, H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = dot(H, PHT) + R
        self.SI = self.inv(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot(PHT, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
