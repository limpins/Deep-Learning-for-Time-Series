% autuanliu@163.com
% 2018年12月10日
% main entry
% 

tic;
clear;
% 读取数据
data = load('./data/linear_signals5D_noise1.mat');      % linear signals
signals = data.linear_signals;
% data = load('./data/nonlinear_signals5D_noise1.mat');   % nonlinear signals
% signals = data.nonlinear_signals;
% data = load('./data/longlag_nonlinear_signals5D_noise1.mat');   % longlag nonlinear signals
% signals = data.longlag_nonlinear_signals;

% 参数设置
norder = 2;
max_lag = 3;
threshold = 3;

%% ----------- RFOLS 估计模型系数 ---------------- %%
[H, Hv] = buildH(signals, norder, max_lag);
[Kalman_H, sparse_H, S, S_No] = term_selector(signals, norder, max_lag, H, threshold);

% 保存重要数据
disp('saving important data ......');
save('./data/linear_terms.mat', 'H', 'Hv', 'Kalman_H', 'sparse_H', 'S', 'S_No');     % linear signals
% save('./data/nonlinear_terms.mat', 'H', 'Hv', 'Kalman_H', 'sparse_H', 'S', 'S_No');  % nonlinear signals
% save('./data/longlag_nonlinear_terms.mat', 'H', 'Hv', 'Kalman_H', 'sparse_H', 'S', 'S_No');  % longlag nonlinear signals
toc;
