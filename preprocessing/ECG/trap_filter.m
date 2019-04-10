% 2019/4/3

function new_ecg = trap_filter(ecg, Fs)
    % 带陷滤波器抑制工频干扰
    % 50Hz陷波器：由一个低通滤波器加上一个高通滤波器组成
    % 而高通滤波器由一个全通滤波器减去一个低通滤波器构成
    % ecg (n_samples x n_ch) 处理单通道信号
    % Fs  采样频率
    %

    Me = 100;                      % 滤波器阶数
    L = 100;                       % 窗口长度
    beta = 100;                    % 衰减系数
    wc1 = 49 / Fs * pi;            % wc1为高通滤波器截止频率，对应 51Hz
    wc2 = 51 / Fs * pi;            % wc2为低通滤波器截止频率，对应 49Hz
    [n_points, n_ch] = size(ecg);

    %% 滤波器定义
    h = ideal_lp(0.132 * pi, Me) - ideal_lp(wc1, Me) + ideal_lp(wc2, Me); % 陷波器
    w = kaiser(L, beta);
    y = h .* rot90(w);       % y为50Hz陷波器冲击响应序列
    %%

    %% 使用滤波器滤波
    new_ecg = zeros(n_points, n_ch);
    for ch=1:n_ch
        new_ecg(:, ch) = filter(y, 1, ecg(:, ch));
    end
    %%

    return;
end

function hd = ideal_lp(wc, Me)
    % 理想低通滤波器
    % 截止角频率wc，阶数Me
    %
    alpha = (Me - 1) / 2;
    n = [0:Me-1];
    p = n - alpha + eps;                % eps为很小的数，避免被0除
    hd = sin(wc * p) ./ (pi * p);       % 用 Sin 函数产生冲击响应
end
