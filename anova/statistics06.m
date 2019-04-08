% 实现 proposition6, 组间关系检验(计算结果部分)
% 在不同的 phrase 下，同一个 组别 之间是否有差异
% email: autuanliu@163.com
% 2017/11/26

% input: 19*20*3 观测值 [pre, ictal, post], 这里会分为不同的组别 O, P, N
clc;
% 这里依赖于之前计算的数据，环境不可以清空，不执行 clear all

% 出度 p值向量 1*3(3 组别)
observations = [outDegreePre, outDegreeIctal, outDegreePost];
% 这里的第三个 返回值 暂时用不着
[outDegreePValue6, outDegreeCompare6, ~] = betweenGroupsTest(observations, 'outDegree', 2);

% 入度 p值向量 1*20
observations = [inDegreePre, inDegreeIctal, inDegreePost];
[inDegreePValue6, inDegreeCompare6, ~] = betweenGroupsTest(observations, 'inDegree', 2);

% 出强度 p值向量 1*20
observations = [outStrengthPre, outStrengthIctal, outStrengthPost];
[outStrengthPValue6, outStrengthCompare6, ~] = betweenGroupsTest(observations, 'outStrength', 2);

% 入强度 p值向量 1*20
observations = [inStrengthPre, inStrengthIctal, inStrengthPost];
[inStrengthPValue6, inStrengthCompare6, ~] = betweenGroupsTest(observations, 'inStrength', 2);

% 清除中间变量
clear observations;

% 图片已经保存，所以这里直接关掉
close all;
