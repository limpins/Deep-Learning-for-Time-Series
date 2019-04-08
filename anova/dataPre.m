% 实现平均,计算出度入度等
% 
% email: autuanliu@163.com
% 2017/11/17

% 为避免对原始数据造成伤害，这里做数据复制
backupPre = PDC_pre_ictal;
backupIctal = PDC_ictal;
backupPost = PDC_post_ictal;

% 平均化处理
avgfreqIctal = avgFreq(backupIctal); 
% 19*20*20 维度数据
avgfreqPost = avgFreq(backupPost);
avgfreqPre = avgFreq(backupPre);
% 至此，获得了3个phrase下的频域上的平均
[Ninterval, Nchannel, ~] = size(avgfreqIctal);

% 0, 1 连通性处理
binaryIctal = avgfreqIctal > threshold;
binaryPre = avgfreqPre > threshold;
binaryPost =avgfreqPost > threshold;
% 至此得到 19*20*20 的3个0,1 二值3维矩阵

% 计算入度
inDegreePre = sum(binaryPre, 2) / Nchannel;
inDegreePre = squeeze(inDegreePre); % 去掉多余维度
inDegreeIctal = sum(binaryIctal, 2) / Nchannel;
inDegreeIctal = squeeze(inDegreeIctal);
inDegreePost = sum(binaryPost, 2) / Nchannel;
inDegreePost = squeeze(inDegreePost);
% 至此得到 19*20列 的 3 个矩阵

% 计算出度
outDegreePre = sum(binaryPre, 3) / Nchannel;
outDegreePre = squeeze(outDegreePre);
outDegreeIctal = sum(binaryIctal, 3) / Nchannel;
outDegreeIctal = squeeze(outDegreeIctal);
outDegreePost = sum(binaryPost, 3) / Nchannel;
outDegreePost = squeeze(outDegreePost);
% 至此得到 19*20行 的 3 个矩阵

% 4 分 连通强度矩阵构造
% input：19*20*20 矩阵
strengthPre = quarterPro(avgfreqPre);
strengthIctal = quarterPro(avgfreqIctal);
strengthPost = quarterPro(avgfreqPost);
% 至此得到 19*20*20 的3个0,0.25,0.50,0.75 四值3维矩阵

% 计算入强度
inStrengthPre = sum(strengthPre, 2) / Nchannel;
inStrengthPre = squeeze(inStrengthPre);
inStrengthIctal = sum(strengthIctal, 2) / Nchannel;
inStrengthIctal = squeeze(inStrengthIctal);
inStrengthPost = sum(strengthPost, 2) / Nchannel;
inStrengthPost = squeeze(inStrengthPost);

% 计算出强度
outStrengthPre = sum(strengthPre, 3) / Nchannel;
outStrengthPre = squeeze(outStrengthPre);
outStrengthIctal = sum(strengthIctal, 3) / Nchannel;
outStrengthIctal = squeeze(outStrengthIctal);
outStrengthPost = sum(strengthPost, 3) / Nchannel;
outStrengthPost = squeeze(outStrengthPost);

% 清除中间变量, 精简变量空间
clear PDC_pre PDC_ictal PDC_post Nchannel Ninterval avgfreqPre avgfreqIctal avgfreqPost;

% 至此所有的预处理和数据准备工作结束