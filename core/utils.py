"""
Email: autuanliu@163.com
Date: 2018/9/28
"""

import datetime as dt

import torch


class Timer():

    def __init__(self):
        pass

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        print(f'Time taken: {end_dt - self.start_dt}')


def set_device(): return torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
