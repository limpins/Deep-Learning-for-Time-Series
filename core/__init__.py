"""
Email: autuanliu@163.com
Date: 2018/9/28
"""

from .loader import *
from .utils import *

torch.backends.cudnn.enabled = True

__all__ = ['Timer', 'set_device', 'get_mat_data', 'get_csv_data',
           'get_txt_data', 'get_excel_data', 'MakeSeqData', 'init_params', 'one_hot_encoding']
