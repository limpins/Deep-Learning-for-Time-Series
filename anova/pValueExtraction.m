% 实现 对数据的提取，主要是 p 值
% email: autuanliu@163.com
% 2017/12/02

outDataDir = '\output\data\compare\';

% 出度
res = pExtract(outDegreeCompare1);
csvwrite([outDataDir, 'outDegreeCompare1.csv'], res);

res = pExtract(outDegreeCompare2);
csvwrite([outDataDir, 'outDegreeCompare2.csv'], res);

res = pExtract(outDegreeCompare4);
csvwrite([outDataDir, 'outDegreeCompare4.csv'], res);

res = pExtract(outDegreeCompare5);
csvwrite([outDataDir, 'outDegreeCompare5.csv'], res);

res = pExtract(outDegreeCompare6);
csvwrite([outDataDir, 'outDegreeCompare6.csv'], res);

res = pExtract(outDegreeCompare7);
csvwrite([outDataDir, 'outDegreeCompare7.csv'], res);

% 入度
res = pExtract(inDegreeCompare1);
csvwrite([outDataDir, 'inDegreeCompare1.csv'], res);

res = pExtract(inDegreeCompare2);
csvwrite([outDataDir, 'inDegreeCompare2.csv'], res);

res = pExtract(inDegreeCompare4);
csvwrite([outDataDir, 'inDegreeCompare4.csv'], res);

res = pExtract(inDegreeCompare5);
csvwrite([outDataDir, 'inDegreeCompare5.csv'], res);

res = pExtract(inDegreeCompare6);
csvwrite([outDataDir, 'inDegreeCompare6.csv'], res);

res = pExtract(inDegreeCompare7);
csvwrite([outDataDir, 'inDegreeCompare7.csv'], res);

% 出强度
res = pExtract(outStrengthCompare1);
csvwrite([outDataDir, 'outStrengthCompare1.csv'], res);

res = pExtract(outStrengthCompare2);
csvwrite([outDataDir, 'outStrengthCompare2.csv'], res);

res = pExtract(outStrengthCompare4);
csvwrite([outDataDir, 'outStrengthCompare4.csv'], res);

res = pExtract(outStrengthCompare5);
csvwrite([outDataDir, 'outStrengthCompare5.csv'], res);

res = pExtract(outStrengthCompare6);
csvwrite([outDataDir, 'outStrengthCompare6.csv'], res);

res = pExtract(outStrengthCompare7);
csvwrite([outDataDir, 'outStrengthCompare7.csv'], res);

% 入强度
res = pExtract(inStrengthCompare1);
csvwrite([outDataDir, 'inStrengthCompare1.csv'], res);

res = pExtract(inStrengthCompare2);
csvwrite([outDataDir, 'inStrengthCompare2.csv'], res);

res = pExtract(inStrengthCompare4);
csvwrite([outDataDir, 'inStrengthCompare4.csv'], res);

res = pExtract(inStrengthCompare5);
csvwrite([outDataDir, 'inStrengthCompare5.csv'], res);

res = pExtract(inStrengthCompare6);
csvwrite([outDataDir, 'inStrengthCompare6.csv'], res);

res = pExtract(inStrengthCompare7);
csvwrite([outDataDir, 'inStrengthCompare7.csv'], res);

% 这里肯定可以使用函数实现，有点懒啊，试了很久，没有做出来
% 但肯定是可以的