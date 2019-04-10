% 2019/4/3
% 参考 https://blog.csdn.net/i_weimoli/article/details/53497384#

function new_ecg = lowband_filter(ecg, Fs)
    % 使用低通滤波器滤除肌电信号
    % ecg (n_samples x n_ch) 处理单通道信号
    % Fs  采样频率
    %
    fp = 80;                                    % 通带截止频率
    fs = 100;                                   % 阻带截止频率
    rp = 1.4;                                   % 通带衰减
    rs = 1.6;                                   % 阻带衰减
    wp = 2 * pi* fp;
    ws = 2 * pi * fs;
    [n_points, n_ch] = size(ecg);

    %% 滤波器设计
    [n, ~] = buttord(wp, ws, rp, rs, 's');      % 巴特沃斯滤波器
    [z, P, k] = buttap(n);                      % 设计归一化巴特沃斯模拟低通滤波器
    [bp, ap] = zp2tf(z, P, k);                  % 转换为 Ha(p)
    [bs, as] = lp2lp(bp, ap, wp);               % Ha(p) 转换为低通Ha(s)并去归一化
    [~, ~] = freqs(bs, as);                     % 模拟滤波器的幅频响应
    [bz, az] = bilinear(bs, as, Fs);            % 对模拟滤波器双线性变换
    [~, ~] = freqz(bz, az);                     % 数字滤波器的幅频响应
    %%

    %% 使用滤波器滤波
    new_ecg = zeros(n_points, n_ch);
    for ch=1:n_ch
        new_ecg(:, ch) = filter(bz, az, ecg(:, ch));
    end
    %%

    return;
end
