% 实现 proposition4, 组内关系检验(计算部分)
% 比较在同一个phrase下，不同的组别 O, P, N 之间的显著性
% email: autuanliu@163.com
% 2017/11/26

% input: 19*20*3 观测值
clc;
% 这里依赖于之前计算的数据，环境不可以清空，不执行 clear all

% outDegree(3个phrase 同时比较, 建立在不同的组别下)
observations = [outDegreePre, outDegreeIctal, outDegreePost];
[outDegreePValue4, outDegreeCompare4, ~] = withinGroupsTest(observations, 'outDegree', 2);

% inDegree(3个phrase 同时比较, 建立在不同的组别下)
observations = [inDegreePre, inDegreeIctal, inDegreePost];
[inDegreePValue4, inDegreeCompare4, ~] = withinGroupsTest(observations, 'inDegree', 2);

% outStrength(3个phrase 同时比较, 建立在不同的组别下)
observations = [outStrengthPre, outStrengthIctal, outStrengthPost];
[outStrengthPValue4, outStrengthCompare4, ~] = withinGroupsTest(observations, 'outStrength', 2);

% inStrength(3个phrase 同时比较, 建立在不同的组别下)
observations = [inStrengthPre, inStrengthIctal, inStrengthPost];
[inStrengthPValue4, inStrengthCompare4, ~] = withinGroupsTest(observations, 'inStrength', 2);

% 清除中间变量
clear observations;

% 图片已经保存，所以这里直接关掉
close all;
