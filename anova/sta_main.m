% autuanliu@163.com
% 2019/04/08
%
clc;
clear all;

% 初始化与设置
format long;
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

% 中值检验
% 这里还会返回一个详细的table信息和各组数据的箱线图，此处全部忽略
[pValueVector.median1, ~, stats] = anova1(WGCI_median1, group, 'on');   % 全部考虑
saveas(gcf, [outputDir, 'median_all_anova.png']);
savefig([outputDir, 'median_all_anova.fig']);
 
[c, medians.median1, ~, gs] = multcompare(stats, 'CType', 'bonferroni', 'display', 'on');
results.median1 = [gs(c(:,1)), gs(c(:,2)), num2cell(c(:, 3:6))];
saveas(gcf, [outputDir, 'median_all.png']);
savefig([outputDir, 'median_all.fig']);

[pValueVector.median2, ~, stats] = anova1(WGCI_median2, group, 'on');   % 只考虑有效系数
saveas(gcf, [outputDir, 'median_part_anova.png']);
savefig([outputDir, 'median_part_anova.fig']);

[c, medians.median2, ~, gs] = multcompare(stats, 'CType', 'bonferroni', 'display', 'on');
results.median2 = [gs(c(:,1)), gs(c(:,2)), num2cell(c(:, 3:6))];
saveas(gcf, [outputDir, 'median_part.png']);
savefig([outputDir, 'median_part.fig']);
% 这里参数如果改为 on，则可以看到交互式图
% 置信水平默认为 0.5，即 alpha = 0.5

% 由于results, means 均是矩阵，所以这里采用 cell 的形式将其保存
% means 是每一个数据列的均值

% 均值检验
[pValueVector.mean1, ~, stats] = anova1(WGCI_mean1, group, 'on');   % 全部考虑
saveas(gcf, [outputDir, 'mean_all_anova.png']);
savefig([outputDir, 'mean_all_anova.fig']);

[c, means.mean1, ~, gs] = multcompare(stats, 'CType', 'bonferroni', 'display', 'on');
results.mean1 = [gs(c(:,1)), gs(c(:,2)), num2cell(c(:, 3:6))];
saveas(gcf, [outputDir, 'mean_all.png']);
savefig([outputDir, 'mean_all.fig']);

[pValueVector.mean2, ~, stats] = anova1(WGCI_mean2, group, 'on');   % 只考虑有效系数
saveas(gcf, [outputDir, 'mean_part_anova.png']);
savefig([outputDir, 'mean_part_anova.fig']);

[c, means.mean2, ~, gs] = multcompare(stats, 'CType', 'bonferroni', 'display', 'on');
results.mean2 = [gs(c(:,1)), gs(c(:,2)), num2cell(c(:, 3:6))];
saveas(gcf, [outputDir, 'mean_part.png']);
savefig([outputDir, 'mean_part.fig']);

% 保存变量
save([outputDir, 'data/', 'workspace_result.mat']);
close all;
