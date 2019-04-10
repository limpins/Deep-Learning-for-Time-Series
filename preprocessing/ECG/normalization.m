% 2019/4/3
function new_ecg = normalization(ecg)
    % ecg 为归一化的 ECG 数据 (n_samples x n_ch)
    %
    new_ecg = mapminmax(ecg.').';
    return;
end
