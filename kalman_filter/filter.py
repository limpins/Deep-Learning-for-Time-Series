"""继承 filterpy.kalman 模块中的 KalmanFilter，并覆盖 update 方法，以满足估计系数的需要，因为在估计系数时需要同时更新 Q 与 R。

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
from filterpy.common import reshape_z
from filterpy.kalman import KalmanFilter
from numpy import dot, eye, zeros


class Linear_Kalman_Estimation(KalmanFilter):
    """定义适用于估计时间序列模型系数的 kalman filter。"""

    def __init__(self, signals, max_lag=3, uc=0.0001):
        """构造函数。

        Args:
            signals (np.array): 可观测信号(n_point*n_dim)
            max_lag (int, optional): Defaults to 3. 自回归模型的阶数，即最大延迟
            uc (float, optional): Defaults to 0.0001. update coefficient, forgetting factor.
        """

        n_row, n_col = signals.shape
        dim_x = max_lag * (n_col ** 2)
        super().__init__(dim_x, n_col, dim_u=0)
        self.max_lag = max_lag
        self.signals = signals
        self.N = n_row     # 信号的长度
        self.ndim = n_col  # 信号的维数
        self.uc = uc
        self.init()

    def init(self):
        self.x = np.random.randn(self.dim_x, 1)         # 初始状态初始化为 (0, 1) 正态分布
        self.Q = self.uc * eye(self.dim_x)              # 文献1的初始化方式，若使用文献3的初始化方式，注释掉该行
        self.H = self.measurement_matrix(self.max_lag)  # 初始时的测量矩阵
        self.z = self.signals[self.max_lag].T           # 初始时的测量值

    def update(self, z, R=None, H=None):
        """与原update函数的唯一不同在于P更新的方式可以调节，为了保证其它功能正常，仍然保留原始的内容"""

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
        elif np.isscalar(R):
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
        # self.P = dot(I_KH, self.P)   # 通常情况下的处理
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def filter(self, t, z):
        """实现一次滤波
        使用 self.x 获取当前的预测值
        
        Args:
            t (int): 当前时间点
            z (np.array): column vector, 当前观测值
        """

        self.predict()
        self.update_Q()
        self.update_R(z)
        self.update_H(t)
        self.update(z.T)

    def forward(self):
        """滤波器的前向操作。
        """

        for time, z in enumerate(self.signals[self.max_lag:]):
            self.filter(time + self.max_lag, z.T)
        return self.x, self.P

    def backward(self):
        """滤波器的后向操作。这里使用同一个滤波器先进行前行操作，之后进行后向操作的连续过程，参看
        smoother 方法，避免使用两个滤波器，在进行后向操作还要使用前向操作的最终状态进行初始化的问题。
        """

        for time, z in enumerate(self.signals[:(self.max_lag-1):-1]):
            self.filter(self.N - 1 - time, z.T)
        return self.x, self.P

    def smoother(self):
        """根据 forward 和 backward 得到的数值进行光滑处理，参考文献 3
        """

        x_f, P_f = self.forward()
        x_b, P_b = self.backward()
        factor1 = self.inv(self.inv(P_f) + self.inv(P_b))
        factor2 = dot(self.inv(P_f), x_f) + dot(self.inv(P_b), x_b)
        x_s = dot(factor1, factor2)
        return x_s

    def measurement_matrix(self, time):
        """计算 C_n, 参考文献3

        Args:
            time: int, 当前的时间点, 从 0 开始

        Returns:
            measurement_matrix: np.array, 与当前时刻对应的转移矩阵
        """

        Yt = lambda t: self.signals[(t-1)::-1, :] if t == self.max_lag else self.signals[(t-1):(t-1-self.max_lag):-1, :]
        Y = Yt(time).reshape(1, -1)
        Cn = np.kron(eye(self.ndim), Y.T)  # 这里其实计算的是 n-1 时刻的 C
        return Cn.T

    def update_R(self, z):
        # update R, 文献 3
        residual = self.residual_of(z)
        self.R = (1 - self.uc) * self.R + self.uc * dot(residual, residual.T)

    def update_Q(self):
        # update Q, 文献 3
        self.Q = self.uc * self._I * np.trace(self.P_prior) / (self.max_lag * self.N)

        # update Q, 文献 1
        # self.Q = self.uc * self._I

    def update_H(self, time):
        # update H, 文献1, 3, 测量矩阵会随时间变化
        self.H = self.measurement_matrix(time)

    def estimate_coef(self, threshold=1e-5):
        """重新排列计算出来的系数(pK^2 x 1)

        Args:
            threshold: 将系数置 0 的阈值, 默认为 1e-5

        Returns:
            y_coef: 以信号形式排列的系数
            A_coef: 以系数矩阵形式排列的系数
        """

        A_coef = []
        x_s = self.smoother()
        x_s[np.abs(x_s) < threshold] = 0.  # 阈值处理

        y_coef = x_s.T.reshape(self.ndim, -1)  # 重新排列系数为 ndim x (ndim * p)
        
        # A_coef 为 p x (ndim x ndim) 形式矩阵
        for m in range(0, y_coef.shape[1], self.ndim):
            A_coef += [y_coef[:, m:(m+self.ndim)]]
        return y_coef, np.array(A_coef)

    def AIC(self):
        """Akaike information criterion.
        A major concern with parametric analysis methods is the order selection of the autoregressive (AR) model. 
        If the order is too small, the frequency content in the data cannot be resolved and the spectral estimates will be biased and smoothed and consequently, 
        some information will be lost. On the other hand, if the model order is overly large, spurious peaks (instabilities) can occur in the spectral estimates, 
        which result in a large variance of error and might be misleading thus causing wrong identification. (reference 3)

        AIC(p) = ln(det(R)) + 2p*d^2 / T

        Returns:
            the AIC value of self.max_lag
        """
     
        aic = lambda p: np.log(np.linalg.det(self.R)) + 2*p*(self.ndim**2)/self.N
        return aic(self.max_lag)

    def BIC(self):
        """Bayesian information criterion.
        p is the model order, T is the length of time series, R is the covariance matrix of the measurement noise, 
        and d is the number of the time series under investigation. Both of these criteria using maximum likelihood principle make a compromise 
        between model complexity and goodness of fit and track both the decreasing error power and the increasing spectral variance with respect to 
        an increasing model order. (reference 3)

        BIC(p) = ln(det(R)) + ln(T)*p*d^2 / T 

        Returns:
            the BIC value of self.max_lag
        """
        
        bic = lambda p: np.log(np.linalg.det(self.R)) + np.log(self.N)*p*(self.ndim**2)/self.N
        return bic(self.max_lag)
