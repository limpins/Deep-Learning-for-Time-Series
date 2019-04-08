% autuanliu@163.com
% 2019/04/08
%
clc;
clear all;

% 初始化与设置
format long;
threshold = 0.1;
inputDir = 'input/';
outputDir = 'output/';
dataName = 'workspace.mat';

% 加载数据
load([inputDir, dataName]);

% 数据截取与备份
% WGCI_mean_low1 = WGCI_mean_low;
% WGCI_mean_mid1 = WGCI_mean_mid(1:15, :, :);
% WGCI_mean_high1 = WGCI_mean_high(1:15, :, :);
% WGCI_median_low1 = WGCI_median_low;
% WGCI_median_mid1 = WGCI_median_mid(1:15, :, :);
% WGCI_median_high1 = WGCI_median_high(1:15, :, :);

% 执行程序
% 中值检验和均值检验
statistics_mean;
statistics_median;

% 保存变量
save([outputDir, 'data/', 'workspace_result.mat'])

% 清除中间变量
clear avg* backup* *Dir;
