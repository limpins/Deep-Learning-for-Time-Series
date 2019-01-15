# 深度学习在EEG数据的应用探索以及实验

```
Author: AutuanLiu
Email: autuanliu@163.com
Date: 2018/11/16
```

**文件夹中与标题同名的PDF文件内容系作者查阅资料总结所得，未经作者允许，请勿擅自传播文档。如有需要请按照上述方式联系作者。**

* 大致目录结构

第一章	时间序列基础
 1.1 时序数据基础
  1.1.1时间序列定义
  1.1.2时间序列分析常见模型
   1.1.2.1 AR模型
   1.1.2.2 MA模型
   1.1.2.3 ARMA模型
   1.1.2.4 MVAR模型
   1.1.2.5 MARX模型
 1.2时序数据的处理
  1.2.1时序数据to监督学习数据
  1.2.2数据标准化/归一化
   1.2.2.1 z-score归一化
   1.2.2.2 min-max归一化
   1.2.2.3 mean 归一化
  1.2.3训练集测试集划分
  1.2.4交叉验证
第二章	深度网络模型
 2.1循环神经网络模型
  2.1.1 RNN模型
   2.1.1.1 RNN的梯度消失问题
   2.1.1.2 双向RNN
  2.1.2 LSTM模型
  2.1.3 GRU模型
  2.1.4 RNN的类型
 2.2 Seq2Seq模型
  2.2.1 编码器
  2.2.2 解码器
 2.3 Attention机制
第三章	时间序列预测任务
 3.1预测的类型
  3.1.1 point-by-point prediction
  3.1.2 multi-sequence prediction
  3.1.3 full sequence prediction
第四章 时间序列分类任务
 4.1分类
第五章DNN在仿真数据上的实验
 5.1 Granger Causality
 5.2 Granger Causality with neural network
 5.3 仿真模型
 5.4 实验结果
第六章DNN在EEG上的应用
 6.1 应用
第七章 参考文献
