clear;clc;
Fs = 256;
O = [1, 2, 4, 5, 7, 8, 12];
P = [6, 10, 11, 15, 19];
N = [3, 9, 13, 14, 16, 17, 18, 20];
load('EEG72s.mat');

% 标准归一化
EEG72s = zscore(EEG72s);

% test 2 channel
ch = [5, 10];

% 2~18s pre-ictal
id1 = [2, 18];
pre_ictal = EEG72s(id1(1)*Fs:id1(2)*Fs, [O, P]);
Pp4Dp1_pre_ictal = EEG72s(id1(1)*Fs:id1(2)*Fs, ch);

% 22~50 seizure
id2 = [22, 50];
seizure = EEG72s(id2(1)*Fs:id2(2)*Fs, [O, P]);

% ictal1 [22, 38]
id3 = [22, 38];
ictal1 = EEG72s(id3(1)*Fs:id3(2)*Fs, [O, P]);
Pp4Dp1_ictal1 = EEG72s(id3(1)*Fs:id3(2)*Fs, ch);

% ictal2 [30, 46]
id4= [30, 46];
ictal2 = EEG72s(id4(1)*Fs:id4(2)*Fs, [O, P]);
Pp4Dp1_ictal2 = EEG72s(id4(1)*Fs:id4(2)*Fs, ch);

% 保存数据
save('seizure.mat', 'seizure');
save('pre_ictal.mat', 'pre_ictal');
save('ictal1.mat', 'ictal1');
save('ictal2.mat', 'ictal2');
save('Pp4Dp1_pre_ictal.mat', 'Pp4Dp1_pre_ictal');
save('Pp4Dp1_ictal1.mat', 'Pp4Dp1_ictal1');
save('Pp4Dp1_ictal2.mat', 'Pp4Dp1_ictal2');
disp('finished!')
