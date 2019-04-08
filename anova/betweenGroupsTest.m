% 实现 proposition1, 组间关系检验(函数设计部分)
% 
% email: autuanliu@163.com
% 2017/11/20

% 为了检验：不同的 phrase 对出度，入度强度等是否有显著性差异
% 每一个 interval 相当于对该 node 的一次抽样
% 这里的数据是均衡数据，即每个 phrase 下的抽样次数是相同的 19 次
% 这里使用 ANOVA 方差分析，假设数据已经满足可以使用 ANOVA 的要求(正态分布等)
% input: 19*20*3 观测值 [pre, ictal, post]
% 为了保存图片，以及区别图片生成(函数执行请求)来源，这里传入一个 flag 参数

function [pValueVector, results, means] = betweenGroupsTest(observations, flag, flag2)
    % 定义目录
    outputDir = 'output\';

    [NInterval, NNode3] = size(observations);
    len = NNode3 / 3;

    % 判断来源
    if flag2 == 1
        % 标签group的生成
        % 1, 2, 3 分别代表 pre, ictal, post
        groups = 1:3;    
        for node = 1:len
            testData = [observations(:, node), observations(:, node + len), observations(:, node + len * 2)];
             % 这里还会返回一个详细的table信息和各组数据的箱线图，此处全部忽略
            [pValueVector(node), ~, stats] = anova1(testData, groups, 'off');
            % 这里参数如果改为 on，则可以看到交互式图
            % 置信水平默认为 0.5，即 alpha = 0.5
            % 更多有关参数设置的帮助，参看 http://cn.mathworks.com/help/stats/anova1.html?searchHighlight=anova1&s_tid=doc_srchtitle

            [results{node}, means{node}, fig, ~] = multcompare(stats, 'CType', 'bonferroni', 'display', 'on');
            % 由于results, means 均是矩阵，所以这里采用 cell 的形式将其保存
            % 这里采用 Bonferroni test 还有其他的矫正方法
            % 更多有关参数设置的帮助，参看 http://cn.mathworks.com/help/stats/multcompare.html?searchHighlight=multcompare&s_tid=doc_srchtitle#bur_iuv

            % 以上两个函数均存在可视化的操作界面，这里我将其关闭了，只保存了具体的结果
            % 第一个函数是进行 ANOVA 方差分析，并返回作为 多重比较的句柄 stats,
            % 第二个函数实现多重比较，也即 post-hoc 的过程

            % 保存交互式比较图像
            saveas(gcf, [outputDir, 'images\proposition1\comparePhrases', flag, num2str(node), '.png']);
        end
    % 包含 flag2 = 2, 3 两种情况
    else
        % 定义组别的长度
        len2 = [0, 6, 4, 10];
        for cata = 1:3

            if flag == 2
                temp = len2(cata + 1);
                % 标签group的生成
                % 1, 2, 3 分别代表 O, P, N
                groups = [ones(1, temp), ones(1, temp) * 2, ones(1, temp) * 3];
                % 这里注意 : 的优先级最低
                index = (1:temp) + len2(cata);
                testData = [observations(:, index), observations(:, index + len), observations(:, index + len * 2)];
            else
                % flag2 == 3 时，单独考虑
                % 平均值后的数据
                % 分别代表 O, P, N
                groups = 1:3;
                % 因为做了平均处理
                len = 3;
                testData = [observations(:, cata), observations(:, cata + len), observations(:, cata + len * 2)];
            end
            
            % 做检验
            [pValueVector(cata), ~, stats] = anova1(testData, groups, 'off');
            [results{cata}, means{cata}, fig, ~] = multcompare(stats, 'CType', 'bonferroni', 'display', 'on');

            % 保存交互式比较图像
            if flag2 == 2
                saveas(gcf, [outputDir, 'images\proposition6\compareCatas', flag, num2str(cata), '.png']);
            else
                saveas(gcf, [outputDir, 'images\proposition7\compareCatas', flag, num2str(cata), '.png']);
            end           
        end
    end
    return;
end