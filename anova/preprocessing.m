% 获取信号的 WGCI 值
% autuanliu@163.com
% 2019/04/08

clc; clear all;
% 初始化与设置
format long;
dataName1 = 'WGCI_mean_persons.mat';
dataName2 = 'WGCI_median_persons.mat';

% 加载数据
load(['input/', dataName1]);
load(['input/', dataName2]);

% 数据值 WGCI_mean_persons、WGCI_median_persons
% 按类型分配结果重新保存
low = [3, 4, 6, 10, 13, 16, 19, 21, 27, 29, 31, 40, 49, 50, 57];
mid = [2, 9, 11, 12, 15, 17, 18, 22, 23, 24, 25, 28, 30, 32, 33, 34, 35, 36, 37, 38, 43, 44, 45, 47, 48, 53, 54, 55, 56, 59, 60, 62, 63, 64];
high = [1, 5, 7, 8, 14, 20, 26, 39, 41, 42, 46, 51, 52, 58, 61, 65, 66, 67, 68, 69];

WGCI_mean_low = WGCI_mean_persons(low, :, :);
WGCI_mean_mid = WGCI_mean_persons(mid, :, :);
WGCI_mean_high = WGCI_mean_persons(high, :, :);

WGCI_median_low = WGCI_median_persons(low, :, :);
WGCI_median_mid = WGCI_median_persons(mid, :, :);
WGCI_median_high = WGCI_median_persons(high, :, :);

% group
group = {1, 69};
for id=low
    group{1, id} = 'low depression';
end

for id=mid
    group{1, id} = 'mid depression';
end

for id=high
    group{1, id} = 'high depression';
end

% 调整数据格式
% 全部考虑
WGCI_mean1 = reshape(WGCI_mean_persons, 69, 9).';
WGCI_median1 = reshape(WGCI_median_persons, 69, 9).';

% 只考虑(0, 1), (1, 0)
WGCI_mean2 = reshape(WGCI_mean_persons(:, [2, 4]), 69, 2).';
WGCI_median2 = reshape(WGCI_median_persons(:, [2, 4]), 69, 2).';

% 清除多余变量
clear low mid high dataName1 dataName2 id

% 保存变量
save(['input/', 'workspace.mat'])
