# 待解决的问题

目前 有几个问题待解决：
1.预测效果不好，学习曲线后面有波折（核心问题）

Epoch 74/80
[1/1] - 1s - train: { MAE (loss): 2.04 - RMSE: 4.37 - MAPE: 9.31% } - val: { MAE: 3.06 - RMSE: 5 - MAPE: 20.3% } - lr: 0.001 - weight decay: 2e-06
Epoch 75/80
[1/1] - 1s - train: { MAE (loss): 2.19 - RMSE: 4.07 - MAPE: 13.1% } - val: { MAE: 2.25 - RMSE: 4.58 - MAPE: 11.9% } - lr: 0.001 - weight decay: 2e-06
Epoch 76/80
[1/1] - 1s - train: { MAE (loss): 2.01 - RMSE: 4.05 - MAPE: 9.94% } - val: { MAE: 2.55 - RMSE: 4.72 - MAPE: 13.5% } - lr: 0.001 - weight decay: 2e-06
Epoch 77/80
[1/1] - 1s - train: { MAE (loss): 2.18 - RMSE: 4.9 - MAPE: 9.7% } - val: { MAE: 2.68 - RMSE: 4.38 - MAPE: 17% } - lr: 0.001 - weight decay: 2e-06
Epoch 78/80
[1/1] - 1s - train: { MAE (loss): 1.94 - RMSE: 3.94 - MAPE: 10.3% } - val: { MAE: 2.81 - RMSE: 6.57 - MAPE: 13.1% } - lr: 0.001 - weight decay: 2e-06
Epoch 79/80
[1/1] - 1s - train: { MAE (loss): 2.42 - RMSE: 5.71 - MAPE: 9.64% } - val: { MAE: 2.54 - RMSE: 5.29 - MAPE: 14.2% } - lr: 0.001 - weight decay: 2e-06
Epoch 80/80
[1/1] - 1s - train: { MAE (loss): 1.94 - RMSE: 4.47 - MAPE: 9.19% } - val: { MAE: 2.93 - RMSE: 5.06 - MAPE: 16.1% } - lr: 0.001 - weight decay: 2e-06

可能要调整参数或者数据结构这类的，或者增加数据量


2.预测性能评价最好用股票相关的，学习一下股票回撤等概念
基础库：
pandas + numpy：计算收益率、波动率等基础指标；
scikit-learn：保留通用指标（如 RMSE）作为基准。
金融专用库：
quantstats：一键生成夏普比率、最大回撤等报告，兼容pandas数据；
backtrader/zipline：回测交易策略，模拟订单执行和成本；
pyfolio：与zipline集成，生成资金曲线、风险分析图表。
可视化工具：
matplotlib/plotly：绘制资金曲线、回撤图、累计收益对比；
networkx：可视化股票相关性网络，验证图结构合理性（如板块聚类）。



# STGNN股票走势预测框架

基于时空图神经网络(STGNN)的股票走势预测框架，将原本用于交通预测的STGNN模型迁移至金融领域，利用其同时捕捉时间关系和空间关系的能力，预测股票市场走势。

## 项目背景

STGNN（Spatial-Temporal Graph Neural Network）最初被设计用于交通流量预测，能够有效处理具有时空特性的数据。在交通预测中，道路上的传感器作为节点采集车辆速度的时间序列数据。

本项目创新性地将STGNN应用于股票走势预测：
- 股票的收盘价类比为交通预测中的"速度"指标
- 不同的股票种类作为图中的节点
- 利用STGNN同时学习股票价格的时间演变规律和不同股票间的空间关联关系

## 安装指南

1. 克隆本项目到本地
```bash
git clone <项目仓库地址>
cd STGNN-Forecasting-Framework
```

2. 安装所需依赖
```bash
pip install -r requirements.txt
```

## 数据说明

项目使用沪深300指数成分股数据，数据组织结构如下：

- `data/csi300/raw/`: 原始数据
  - `adj_mx_csi300.pkl`: 股票间的邻接矩阵
  - `adj_mx_stocks_gaussian_kernel.pkl`: 基于高斯核的股票关联矩阵
  - `csi300.h5`: 原始股票数据
  - `distances_csi300.csv`: 股票间的"距离"度量

- `data/csi300/processed/`: 处理后的数据集
  - 包含训练集、验证集和测试集
  - `scaler.pkl`: 数据标准化器

- `data/csi300/predicted/`: 模型预测结果

- `data/csi300/structured/`: 结构化数据
  - `node_locations.pkl`: 节点位置信息

## 模型说明

本项目实现的STGNN模型能够同时捕捉：
- 时间维度：股票价格随时间的变化趋势
- 空间维度：不同股票之间的关联性和影响

模型代码位于`src/spatial_temporal_gnn/`目录下，已训练好的模型 checkpoint 存储在`models/checkpoints/st_gnn_csi300.pth`。

## 使用方法

### 模型训练与验证

可以通过Jupyter Notebook进行模型训练与验证：
```bash
jupyter notebook Analysis/STGNN模型训练与验证.ipynb
```

### 数据处理

数据处理相关脚本位于`src/data/`目录，包括：
- 数据提取 (`data_extraction.py`)
- 数据处理 (`data_processing.py`)
- 数据集构建 (`dataset_builder.py`)
- 数据加载器 (`dataloaders.py`)

## 项目结构

```
STGNN-Forecasting-Framework/
├── Analysis/                 # 分析与可视化
│   └── STGNN模型训练与验证.ipynb  # 模型训练与验证笔记本
├── data/                     # 数据目录
│   └── csi300/               # 沪深300相关数据
│       ├── predicted/        # 预测结果
│       ├── processed/        # 处理后的数据集
│       ├── raw/              # 原始数据
│       └── structured/       # 结构化数据
├── models/                   # 模型相关
│   └── checkpoints/          # 模型 checkpoint
├── src/                      # 源代码
│   ├── data/                 # 数据处理模块
│   ├── explanation/          # 解释模块
│   ├── spatial_temporal_gnn/ # STGNN模型实现
│   ├── utils/                # 工具函数
│   └── verbal_explanations/  # 文字解释生成
├── README.md                 # 项目说明文档
└── requirements.txt          # 依赖列表
```

## 结果分析

模型预测结果存储在`data/csi300/predicted/`目录，可以通过`Analysis`目录下的Jupyter Notebook进行结果可视化和性能评估。

## 参考资料

1. STGNN相关原始论文：[Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting](https://arxiv.org/abs/1709.04875)
2. 沪深300指数相关资料：[AKShare 财经数据接口库](https://www.akshare.xyz/)