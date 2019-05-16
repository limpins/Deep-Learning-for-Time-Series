import pickle
import pandas as pd
import scipy.io as sio
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ujson

data_root = Path(r'depression/')
WGCI_persons, med_values, med_type_presons = {}, {}, {}
# file_name = rf'./configs/depression.json'
file_name = rf'./configs/depression_no_overlap.json'
# file_name = rf'./configs/depression512.json'
cfg = ujson.load(open(file_name, 'r'))
sta_type = 'max'  # 'median', 'mean

# 获取哈密顿分数信息
data = pd.read_csv(r'Data/depression/info.csv')
scores = {data.loc[i, 'subjects']: data.loc[i, 'score'] for i in range(data.shape[0])}
ret = sorted(scores.items(), key=lambda x: x[1])

# 保存结果
with open(data_root / r'result/scores_persons.pkl', 'wb') as outfile:
    pickle.dump(ret, outfile)


# 保存结果
with open(data_root/r'result/scores.pkl', 'wb') as outfile:
    pickle.dump(scores, outfile)

for id in range(1, 70):
    data = np.loadtxt(data_root / f'WGCI_{id}.txt', dtype=np.float32, skiprows=1, comments='#')
    WGCI_trials = data.reshape(cfg['trials'], 3, 3)
    WGCI_persons[id] = WGCI_trials
    med_WGCI_persons = getattr(np, sta_type)(WGCI_trials, 0)  # 中值或均值或最大值
    med_values[id] = med_WGCI_persons

# 保存结果
with open(data_root/ r'result/WGCI_persons.pkl', 'wb') as outfile:
    pickle.dump(WGCI_persons, outfile)

# 保存结果
with open(data_root/ rf'result/WGCI_{sta_type}_persons.pkl', 'wb') as outfile:
    pickle.dump(med_values, outfile)


# mat 格式
WGCI_persons1 = []
for id in range(1, 70):
    WGCI_persons1.append(med_values[id])
WGCI_persons1 = np.array(WGCI_persons1)
sio.savemat(data_root / rf'result/WGCI_{sta_type}_persons.mat', {'WGCI_persons': WGCI_persons1})
sio.savemat(f'./anova/input/WGCI_{sta_type}_persons.mat', {f'WGCI_{sta_type}_persons': WGCI_persons1})


# 读取结果
# with open(data_root/ rf'result/WGCI_persons.pkl', 'rb') as instream:
#     data = pickle.load(instream)
#     print(data)

# 读取结果
# with open(data_root/ rf'result/WGCI_{sta_type}_persons.pkl', 'rb') as instream:
#     data = pickle.load(instream)
#     print(data)

# 每种患者的 WGCI 中值

for type in ['low', 'mid', 'high']:
    type_med = np.zeros((len(cfg[type]), 3, 3))
    for id, idx in zip(cfg[type], range(len(cfg[type]))):
        type_med[idx] = med_values[id]
    med_type_presons[type] = getattr(np, sta_type)(type_med, 0)

# 保存结果
with open(data_root / rf'result/WGCI_{sta_type}_type_persons.pkl' , 'wb') as outstream:
    pickle.dump(med_type_presons, outstream, 0)

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

# 读取结果
# with open(data_root / r'result/patient_label.pkl' , 'rb') as instream:
#     type_label = pickle.load(instream)
