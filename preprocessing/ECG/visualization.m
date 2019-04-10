% 2019/4/3
function visualization(origin_ecg, new_ecg, Fs, savename)
    % origin_ecg  原始数据
    % new_ecg  处理过后的数据
    % Fs 采样频率
    % savename  保存的文件名
    %
    %% 可视化处理结果 只查看其中一个通道
    n_points = size(new_ecg, 1);
    TIME = (1:n_points) / Fs;    % 时间信息

    figure(1);
    subplot (2, 1, 1);
    plot(TIME, origin_ecg(:, 1));
    xlabel('t(s)');
    ylabel('ECG(mV)');
    title('Origin ECG');
    grid on;

    subplot(2, 1, 2);
    plot(TIME, new_ecg(:, 1));
    xlabel('t(s)');
    ylabel('ECG(mV)');

    title('Processed ECG');
    grid on;

    % 保存可视化结果
    saveas(1, savename);
    close all;
    %%
end
