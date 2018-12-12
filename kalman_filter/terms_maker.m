% autuanliu@163.com
% 2018年12月10日
% main entry
% 

tic;
clear;

flag='longlag_nonlinear'; % !set{'linear', 'nonlinear', 'longlag_nonlinear'}

% 读取数据
switch flag
    case 'linear'
        norder = 1;
        data = load('./data/linear_signals5D_noise1.mat');      % linear signals
        signals = data.linear_signals;
    case 'nonlinear'
        norder = 2;
        data = load('./data/nonlinear_signals5D_noise1.mat');   % nonlinear signals
        signals = data.nonlinear_signals;
    case 'longlag_nonlinear'
        norder = 2;
        data = load('./data/longlag_nonlinear_signals5D_noise1.mat');   % longlag nonlinear signals
        signals = data.longlag_nonlinear_signals;
    otherwise
        disp('not define!')
end

% 参数设置
max_lag = 5;
threshold = 5;

%% ----------- RFOLS 估计模型系数 ---------------- %%
[H, Hv] = buildH(signals, norder, max_lag);
[Kalman_H, sparse_H, S, S_No] = term_selector(signals, norder, max_lag, H, threshold);

% 保存重要数据
disp('saving important data ......');
switch flag
    case 'linear'
        save('./data/linear_terms.mat', 'H', 'Hv', 'Kalman_H', 'sparse_H', 'S', 'S_No');     % linear signals
    case 'nonlinear'
        save('./data/nonlinear_terms.mat', 'H', 'Hv', 'Kalman_H', 'sparse_H', 'S', 'S_No');  % nonlinear signals
    case 'longlag_nonlinear'
        save('./data/longlag_nonlinear_terms.mat', 'H', 'Hv', 'Kalman_H', 'sparse_H', 'S', 'S_No');  % longlag nonlinear signals
    otherwise
        disp('not define!')
end
toc;
