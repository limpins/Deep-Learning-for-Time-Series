% 实现 proposition2, 组内关系检验(函数设计部分)
% 
% email: autuanliu@163.com
% 2017/11/25

% 为了检验：同一个 phrase 下，不同的 node 在出度，入度强度等是否有显著性差异
% 每一个 node 都有 19 个值
% 这里的数据是均衡数据，即每个 node 下的抽样次数是相同的 19 次
% 这里使用 ANOVA 方差分析，假设数据已经满足可以使用 ANOVA 的要求(正态分布等)
% post-hoc 使用 Bonferroni test
% input: 19*20*3 例如 出度 的值 * 3 个phrase
% 为了保存图片，以及区别图片生成(函数执行请求)来源，这里传入一个 flag 参数

% output:
% pValueVector: 3 个 p 值 分别代表 pre, ictal, post
% result: 3 cell*(190*6) (190 个对应的 p 值)
% means: 各组的均值 3 cell * 20 * 2(每组的平均值， 标准差)

function [pValueVector, results, means] = withinGroupsTest(observations, flag, flag2)
    % 定义目录
    outputDir = 'output\';
    [NInterval, NNode] = size(observations);
    len = NNode / 3;

    % 判断位置，确定 groups
    if flag2 == 1
        % 标签group的生成
        % 1, 2, 3, ... 分别代表 node1, node2, node3, ...
        groups = 1:20;
    elseif flag2 == 2
        % 1, 2, 3 分别代表 O, P, N
        groups = [ones(1, 6), ones(1, 4) * 2, ones(1, 10) * 3];
    else
        groups = 1:3;
        % 因为做了平均化处理，所以长度变为 3
        len = 3;
    end

    % 共有 3 个时期
    for phrase = 1:3 
        % 这里还会返回一个详细的table信息和各组数据的箱线图，此处全部忽略
        [pValueVector(phrase), ~, stats] = anova1(observations(:, groups + len * (phrase - 1)), groups, 'off');
        % 这里参数如果改为 on，则可以看到交互式图
        % 置信水平默认为 0.5，即 alpha = 0.5

        [results{phrase}, means{phrase}, fig, ~] = multcompare(stats, 'CType', 'bonferroni', 'display', 'on');
        % 由于results, means 均是矩阵，所以这里采用 cell 的形式将其保存
        % means 是每一个数据列的均值
   
        % 判断存储位置
        if flag2 == 1
            % 保存交互式比较图像
            saveas(gcf, [outputDir, 'images\proposition2\compareNodes', flag, num2str(phrase), '.png']);
        elseif flag2 == 2
            saveas(gcf, [outputDir, 'images\proposition4\compareCatas', flag, num2str(phrase), '.png']);
        else
            saveas(gcf, [outputDir, 'images\proposition5\compareCatasAvg', flag, num2str(phrase), '.png']);
        end       
    end
    return;
end