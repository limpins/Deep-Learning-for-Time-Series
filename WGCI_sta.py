import pickle
import scipy.io as sio
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ujson

data_root = Path(r'depression/')
WGCI_persons, med_values, med_type_presons = {}, {}, {}
sta_type = 'mean' # 'median'

for id in range(1, 70):
    data = np.loadtxt(data_root / f'WGCI_{id}.txt', dtype=np.float32, skiprows=1, comments='#')
    WGCI_trials = data.reshape(100, 3, 3)
    WGCI_persons[id] = WGCI_trials
    med_WGCI_persons = getattr(np, sta_type)(WGCI_trials, 0)  # 中值或均值
    med_values[id] = med_WGCI_persons

# 保存结果
with open(data_root/ r'result/WGCI_persons.pkl', 'wb') as outfile:
    pickle.dump(WGCI_persons, outfile, 0)

# mat 格式
sio.savemat(data_root/ r'result/WGCI_persons.mat', {'WGCI_persons': WGCI_persons})

# 保存结果
with open(data_root/ rf'result/WGCI_{sta_type}_persons.pkl', 'wb') as outfile:
    pickle.dump(med_values, outfile, 0)

# mat 格式
sio.savemat(data_root/ rf'result/WGCI_{sta_type}_persons.mat', {'med_values': med_values})

# 读取结果
# with open(data_root/ rf'result/WGCI_persons.pkl', 'rb') as instream:
#     data = pickle.load(instream)
#     print(data)

# 读取结果
# with open(data_root/ rf'result/WGCI_{sta_type}_persons.pkl', 'rb') as instream:
#     data = pickle.load(instream)
#     print(data)

# 每种患者的 WGCI 中值
file_name = rf'./configs/depression.json'
cfg = ujson.load(open(file_name, 'r'))

for type in ['low', 'mid', 'high']:
    type_med = np.zeros((len(cfg[type]), 3, 3))
    for id, idx in zip(cfg[type], range(len(cfg[type]))):
        type_med[idx] = med_values[id]
    med_type_presons[type] = getattr(np, sta_type)(type_med, 0)

# 保存结果
with open(data_root / rf'result/WGCI_{sta_type}_type_persons.pkl' , 'wb') as outstream:
    pickle.dump(med_type_presons, outstream, 0)

# mat 格式
sio.savemat(data_root / rf'result/WGCI_{sta_type}_type_persons.mat', {'med_type_presons': med_type_presons})

# 读取结果
# with open(data_root / rf'result/WGCI_{sta_type}_type_persons.pkl' , 'rb') as instream:
#     med_type_presons = pickle.load(instream)


# 患者类型标签
type_label = {}
for p_type in ['low', 'mid', 'high']:
    for id in cfg[p_type]:
        type_label[id] = p_type

# 保存结果
with open(data_root / r'result/patient_label.pkl' , 'wb') as outstream:
    pickle.dump(type_label, outstream, 0)

# mat 格式
sio.savemat(data_root / r'result/patient_label.mat', {'type_label': type_label})

# 读取结果
# with open(data_root / r'result/patient_label.pkl' , 'rb') as instream:
#     type_label = pickle.load(instream)
