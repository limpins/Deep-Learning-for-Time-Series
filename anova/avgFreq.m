% script 说明
% 实现对 freq 进行平均
% 原始数据维度：19*20*20*128
% email: autuanliu@163.com
% 2017/11/17

function output = avg_freq(input)
    % 输入输出都是 高维矩阵
    output = mean(input, 4);
    % 去掉多余的维度
    output = squeeze(output);
    return;
end