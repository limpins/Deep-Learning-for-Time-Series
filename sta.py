import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

data_root = Path(r'depression/')
med_values = []
med_low = []
med_mid = []
med_high = []

for id in range(1, 70):
    x = np.loadtxt(
        data_root / f'WGCI_{id}.txt', dtype=float, skiprows=1, comments='#')
    a = x.reshape(100, 3, 3)
    med = np.median(a, 0)
    # print(med)
    # print('hello')
    med_values.append(med)


"""
    with open('均值.txt','w') as outfile:
        outfile.write(f'# Array shape :{100,3,3}\n')
        for med_values_slice in med_values:
            np.savetxt(outfile,med_values_slice,fmt='%.6f')
            outfile.write('# New trial\n')
    # 这里使用我给你的写 3D array的函数，写到一个 txt 文件中
    # file_handle = open("均值.txt", mode='w')
    # file_handle.write('中值矩阵为 \n')
    # np.savetxt("均值.txt", med)
    # df = pd.DataFrame()
    # for i in range(9):
    #     df[i] = list_all[i]
    # plt.boxplot(x=df.values, labels=df.columns, whis=0.005)
    data = a.reshape(100, -1)
    plt.boxplot(x=data, labels=list(list(map(str, range(9)))), whis=0.005)
    plt.savefig(f"第{id}个病人的箱线图.png")
    # plt.show()
"""
#med_values = np.asarray(med_values)

low = [3, 4, 6, 10, 13, 16, 19, 21, 27, 29, 31, 40, 49, 50, 57]
mid = [2, 9, 11, 12, 15, 17, 18, 22, 23, 24, 25, 28, 30, 32, 33, 34, 35, 36, 37, 38, 43, 44, 45, 47, 48, 53, 54, 55, 56, 59, 60, 62, 63, 64]
high = [1, 5, 7, 8, 14, 20, 26, 39, 41, 42, 46, 51, 52, 58, 61, 65, 66, 67, 68, 69]
for idx in low:
    med_low.append(med_values[idx-1])
for idx in mid:
    med_mid.append(med_values[idx-1])
for idx in high:
    med_high.append(med_values[idx-1])
low = np.median(med_low,0)
mid = np.median(med_mid,0)
high = np.median(med_high,0)
print(low)
print(mid)
print(high)
med_low = np.asarray(med_low)
med_mid = np.asarray(med_mid)
med_high = np.asarray(med_high)
data_low = med_low.reshape(15, -1)
data_mid = med_mid.reshape(34, -1)
data_high = med_high.reshape(20, -1)
plt.boxplot(x = data_low,labels=list(list(map(str, range(9)))),whis=0.005)
plt.savefig("第一种病人的箱线图.png")
plt.boxplot(x=data_mid, labels=list(list(map(str, range(9)))), whis=0.005)
plt.savefig("第二种病人的箱线图.png")
plt.boxplot(x=data_high, labels=list(list(map(str, range(9)))), whis=0.005)
plt.savefig("第三种病人的箱线图.png")

