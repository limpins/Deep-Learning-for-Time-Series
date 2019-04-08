% 实现 连通性4值处理
% 原始数据维度：19*20*20(经过频率平均处理)
% 这个函数用于实现对 PDC 值的 4 值划分 0, 0.25, 0.50, 0.75
% input: 一个 3 维矩阵, Ninterval * Nchannel * Nchannel
% output: 返回经过 4 值处理的 3 维矩阵
% email: autuanliu@163.com
% 2017/11/20

function output = quarterPro(input)
    [Ninterval, Nchannel, ~] = size(input);

    for index1 = 1:Ninterval
        for index2 = 1:Nchannel
            for index3 = 1:Nchannel
                x = input(index1, index2, index3);
                % 幅值不可能 < 0 ,这里不考虑 < 0 的情况
                if x <= 0.25
                    output(index1, index2, index3) = 0;
                elseif x <= 0.5
                    output(index1, index2, index3) = 0.25;
                elseif x <= 0.75
                    output(index1, index2, index3) = 0.50;
                else
                    output(index1, index2, index3) = 0.75;
                end
            end
        end
    end
    return;
end