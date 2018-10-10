"""
Email: autuanliu@163.com
Date: 2018/9/28
"""

from torch.utils.data import Dataset
import scipy.io as sio
import pandas as pd


class Data(Dataset):
    def __init__(self, file_name):
        super(Data, self).__init__()
        pass

    def get_mat_data(self):
        pass

    def get_csv_data(self):
        pass

    def get_txt_data(self):
        pass

    def get_excel_data(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
