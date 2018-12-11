% autuanliu@163.com
% 2018年12月10日
% 

NN = 5030;
ndim = 3; % 信号的维度
fs = 256;          % Ts 采样周期 fs 采样频率
N = 1000;          % 用于真是计算的数据点的长度
M = 2; % 子系统的个数
NL_num = 2;
ts = (1:NN)/fs;    % 时域中采样的相对时间点  
f = (1:NN)*fs/NN; % 频域中采样的频率点
y1 = zeros(NN, 1); % 声明数据的维度
y2 = zeros(NN, 1);
y3 = zeros(NN, 1);
noise1 = randn(NN, 1);  % 生成随机噪音数据
noise2 = randn(NN, 1);
noise3 = randn(NN, 1);
lags = [2, 2, 2];  % 分别对应于各个信号的延迟 lag
norder = 2;        % 非线性最高次数(order)
threshold = 0.99999;  % 进行 FROLS 估计系数时的阈值
err_std = 1;


% 生成仿真信号
%% ----------- 非线性模型  生成y1、y2、y3信号数据 ----------- %%
% 为保证信号延迟有意义，前 lag 信号点使用随机数据
y1(1:lags(1)) = rand(lags(1), 1);
y2(1:lags(2)) = rand(lags(2), 1);
y3(1:lags(3)) = rand(lags(3), 1);

% 生成标准差为 err_std 的白噪声
noise1 = (noise1 - mean(noise1))/std(noise1) * err_std;
noise2 = (noise2 - mean(noise2))/std(noise2) * err_std;
noise3 = (noise3 - mean(noise3))/std(noise3) * err_std;

for n=3:NN  % 信号时域
    y1(n) = 0.5 * y3(n-1);
    y2(n) = 0.5 * y2(n-1) - 0.3 * y2(n-2) + 0.1 * y3(n-2) + 0.4 * y3(n-1) * y3(n-2);
    y3(n) = 0.3 * y3(n-1) - y3(n-2) - 0.1 * y2(n-2);
end

signals = [y1, y2, y3];

%% ----------- RFOLS 估计模型系数 ---------------- %%
H = buildH(signals, 2, 3);
[sparse_H, S, S_No] = term_selector(signals, 2, 3, H, threshold);

% 保存重要数据
% disp('saving important data ......');
% save('signals_data.mat', 'signals', 'coefs', 'errs', 'y_spectrums', 'err_spectrums');
