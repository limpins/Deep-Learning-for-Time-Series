% Email: autuanliu@163.com
% Date: 2018/10/10
% 生成 仿真数据
%
% 初始化设置
npoint = 2048;     % 待研究或者采样的信号长度
nlen = 2100;       % 仿真信号的总长度
nchannel = 2;       % 信号的维度
max_lag = 20;       % 最大时延
err_var = 1;        % 噪音的方差
err_mean = 0;       % 噪音的均值
noise = make_noise(nlen, nchannel, err_mean, err_var, flag);
init = init_signal(max_lag, nchannel);
x1(1:max_lag, 1) = init(:, 1) + noise(1:max_lag , 1);
x2(1:max_lag, 1) = init(:, 2) + noise(1:max_lag , 2);


% 非线性信号
for t=(max_lag + 1):nlen  % 信号时域
    x1(t) = 0.95*sqrt(2) * x1(t-1) - 0.9025 * x1(t-2) + noise(t, 1);
    x2(t) = 0.5 * x1(t-2) * x1(t-2) + noise(t, 2);
end

% 设置非线性信号并保存仿真数据
nonlinear_signals = [x1, x2];
nonlinear_signals = nonlinear_signals((max_lag+1):(max_lag+npoint), :);
% 含有噪音
save('nonlinear_signals.mat', 'nonlinear_signals');

% 长时延非线性信号
for t=(max_lag + 1):nlen  % 信号时域
    x1(t) = 0.95*sqrt(2) * x1(t-1) - 0.9025 * x1(t-2) + noise(t, 1);
    x2(t) = 0.5 * x1(t-10) * x1(t-10) + noise(t, 2);
end

% 设置长时延非线性信号并保存仿真数据
longlag_nonlinear_signals = [x1, x2];
longlag_nonlinear_signals = longlag_nonlinear_signals((max_lag+1):(max_lag+npoint), :);
% 含有噪音
save('longlag_nonlinear_signals.mat', 'longlag_nonlinear_signals');


function noise = make_noise(npoint, nchannel, mean_v, variance, flag)
    % flag == 0 表示不加噪音
    if flag == 0
        noise = zeros(npoint, nchannel);
    else
        noise = randn(npoint, nchannel);
        noise = (noise - mean(noise))./std(noise);
        if variance ~= 0
            noise = mean_v + noise * sqrt(variance);
        end
    end
    return;
end

function init = init_signal(max_lag, nchannel)
    init = randn(max_lag, nchannel);
    return;
end
