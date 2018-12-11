"""基于 FROLS 算法的候选项选择器算法(通用算法，无关线性或者非线性)

**候选项的排列顺序所遵从的主要原则：
1. 线性项在前，非线性项在后
2. 先考虑简单形式再考虑复杂形势
3. 统一取最大延迟
4. 具体实现思路参看 [<<P 矩阵的生成算法.md>>](https://gitlab.com/AutuanLiu/Epilepsy-disease-research/blob/master/theory/P矩阵生成算法.md)

3 个信号或通道，非线性次数为 2，则具体的排列顺序为                 {共27项}
**线性项:   
y1(t-1), y1(t-2), y2(t-1), y2(t-2), y3(t-1), y3(t-2)             {6}

**非线性项: 
y1^2(t-1), y1(t-1)y1(t-2), y1^2(t-2);                            {3}
y1(t-1)y2(t-1), y1(t-1)y2(t-2), y1(t-2)y2(t-1), y1(t-2)y2(t-2);  {4}
y1(t-1)y3(t-1), y1(t-1)y3(t-2), y1(t-2)y3(t-1), y1(t-2)y3(t-2);  {4}
y2^2(t-1), y2(t-1)y2(t-2), y2^2(t-2);                            {3}
y2(t-1)y3(t-1), y2(t-1)y3(t-2), y2(t-2)y3(t-1), y2(t-2)y3(t-2);  {4}
y3^2(t-1), y3(t-1)y3(t-2), y3^2(t-2);                            {3}


**参考文献：
1. Billings S A, Chen S, Korenberg M J. Identification of MIMO non-linear systems using a forward-regression orthogonal estimator[J]. International Journal of Control, 1989, 49(6):2157-2189.
2. Billings S A. Nonlinear system identification : NARMAX methods in the time, frequency, and spatio-temporal domains[M]. Wiley, 2013.

Copyright:
----------
    Author: AutuanLiu
    Date: 2018/12/10
"""

import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    # define device


class Selector:
    """基于 FROLS 算法的候选项选择器算法(通用算法，无关线性或者非线性)

    Attributes:
        signals (np.array or torch.Tensor): N*Ndim 模型信号
        norder (int): 非线性次数
        max_lag (int): max lag.
        threshold (float): 停止迭代的阈值
        N (int): the length of signals.
        ndim (int): the channel or dim of signals.
    """

    def __init__(self, signals, norder, max_lag, threshold=0.99999):
        """基于 FROLS 算法的候选项选择器算法
        
        Args:
            signals (np.array or torch.Tensor): N*Ndim 模型信号
            norder (int): 非线性次数
            max_lag (int): max lag.
            threshold (float, optional): Defaults to 0.99999. 停止迭代的阈值
        """

        self.signals = torch.as_tensor(signals, dtype=torch.float, device=device)
        self.N, self.ndim = self.signals.shape
        self.norder = norder
        self.max_lag = max_lag
        self.threshold = threshold
        self.lags_sum = self.ndim * max_lag

    def buildV(self, n_cnt):
        """生成 V 矩阵
        
        Args:
            n_cnt (int): 当前的非线性次数(包括1)且 n_cnt >= 1
        
        Returns:
            V (torch.Tensor): float type, V.int()
        """

        V = []
        if n_cnt < 1:
            raise ValueError(f'n_cnt >= 1 while n_cnt = {n_cnt}')
        elif n_cnt == 1:
            V = 1. + torch.arange(self.ndim * self.max_lag).view(-1, 1).float()
        else:
            V1 = self.buildV(n_cnt - 1)
            print(V1)
            nrow = V1.size(0)
            for t in range(1, self.lags_sum + 1):
                idx = torch.where(V1[:, 0].int() == t, torch.arange(nrow), torch.tensor(nrow + 1)).min().item()
                print(idx)
                print(nrow - idx + 1)
                V2 = torch.stack([torch.ones(nrow - idx, 1) * t, V1[idx:(nrow+1), :]], dim=1).squeeze()
                V.append(V2)
            print(V)
        return torch.tensor(V)
