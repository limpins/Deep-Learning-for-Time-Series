% 有必要对原始数据进行对数变换并最大最小归一化
%normalization - 对数据进行 min-max 归一化
% 对每一行进行归一化，输入数据为 n_samples x n_features
% mins 最小值
% maxs 最大值
%

% 数据路径
root = './depression/';
% 归一化范围 [-1, 1]
mins = -1;
maxs = 1;

for id=1:69
    name = [root, int2str(id), '.mat'];
    load(name);
    data = mapminmax(MK(:, 2:end).', mins, maxs).';
    save([root, int2str(id), 'normalized.mat'], 'data');
end
