# Wind-10m: 10米风速机器学习预报校正系统

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3%2B-green)]()

基于 LightGBM 的 ECMWF 数值天气预报 10 米风速后处理校正系统。

## 功能特性

- **数据预处理**：从 EC4AI 提取 30 个气象要素到 97 个国家气象观测站
- **机器学习训练**：LightGBM 多策略训练（统一模型/分时效/分站点）
- **模型评估**：多维度评分（R/ME/MAE/RMSE/TS/ETS）
- **后处理技术**：AVT、PDFM、EMD、OTS 等

## 项目结构

```
wind_10m/
├── data/                   # 数据目录
├── preprocessing/          # 数据预处理
│   ├── extract_ec.py      # EC4AI → CSV
│   └── convert_npy.py     # CSV → npy
├── models/                 # 模型训练脚本
│   ├── LightGBM.py        # 主训练脚本
│   ├── cnn_demo.py        # CNN 示例
│   └── ...
├── evaluation/             # 模型评估
│   ├── access97.py        # 97站点评估
│   └── access97_uv.py     # UV分量评估
├── src/                    # 工具模块
│   └── utils.py
└── lgb_predict.py         # 预测脚本
```

## 快速开始

```bash
# 1. 提取 EC 数据
python preprocessing/extract_ec.py

# 2. 转换数据格式
python preprocessing/convert_npy.py

# 3. 训练模型
python models/LightGBM.py

# 4. 评估效果
python evaluation/access97.py
```

## 技术栈

- Python 3.8+
- LightGBM / scikit-learn
- numpy / pandas / arrow
- meteva（气象检验库）

## 作者

- **Author**: yinlb
- **Created**: 2023-07
- **Modified**: 2026-04-03

详见 [AGENTS.md](AGENTS.md) 完整文档。
