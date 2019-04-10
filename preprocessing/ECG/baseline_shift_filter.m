% 2019/4/3
function new_ecg = baseline_shift_filter(ecg, Fs)
    % IIR零相移数字滤波器纠正基线漂移
    % ecg (n_samples x n_ch) 处理单通道信号
    % Fs  采样频率
    %

    Wp = 1.4 * 2 / Fs;                            % 通带截止频率
    Ws = 0.6 * 2 / Fs;                            % 阻带截止频率
    devel = 0.005;                                % 通带纹波
    Rp = 20 * log10((1 + devel) / (1 - devel));   % 通带纹波系数
    Rs = 20;                                      % 阻带衰减
    [n_points, n_ch] = size(ecg);

    %% 构建滤波器
    [N, Wn] = ellipord(Wp, Ws, Rp, Rs, 's');      % 求椭圆滤波器的阶次
    [b, a] = ellip(N, Rp, Rs, Wn, 'high');        % 求椭圆滤波器的系数
    [~, ~] = freqz(b, a, 512);
    %%

    %% 使用滤波器滤波
    new_ecg = zeros(n_points, n_ch);
    for ch=1:n_ch
        new_ecg(:, ch) = filter(b, a, ecg(:, ch));
    end
    %%

    return;
end
