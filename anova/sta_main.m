% 获取信号的 PDC 值, 使用基于 Kalman Filter 的参数估计
% autuanliu@163.com
% 2018/03/24
clc;
clear all;

% 初始化与设置
format long;
threshold = 0.1;
inputDir = 'input/';
outputDir = 'output/';
dataName = 'PDC_pre_ictal.mat';
dataName1 = 'PDC_ictal.mat';
dataName2 = 'PDC_post_ictal.mat';

% 加载数据
PDC_pre_ictal = load([inputDir, dataName]);
PDC_pre_ictal = PDC_pre_ictal.PDC_mean;
PDC_ictal = load([inputDir, dataName1]);
PDC_ictal = PDC_ictal.PDC_mean;
PDC_post_ictal = load([inputDir, dataName2]);
PDC_post_ictal = PDC_post_ictal.PDC_mean;

% 执行程序
dataPre;
statistics04;
statistics06;

% 保存变量
save([outputDir, 'data/', 'workspace.mat'])

% 清除中间变量
clear avg* backup* *Dir;
