% 实现 对数据的提取，主要是 p 值, flag=1, 2 分别对应着
% 有 20 个值和3个值的情况
% email: autuanliu@163.com
% 2017/12/02

function res = pExtract(input)
        len = length(input);
        res = input{1}(:, [1, 2, 6]);
        for index = 2:len
            res(:, index + 2) = input{index}(:, 6);
        end
    return;
end