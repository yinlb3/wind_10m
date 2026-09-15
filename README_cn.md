# Wind-10m

> 基于 ECMWF 数值天气预报产品的 10 米风速机器学习预报校正系统

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3%2B-green)]()

基于 LightGBM 的 ECMWF 10 米风速预报后处理与偏差校正系统。

## 目录

- [安装](#安装)
- [使用](#使用)
- [项目结构](#项目结构)
- [里程碑](#里程碑)
- [许可证](#许可证)
- [作者](#作者)

## 安装

需要 Python 3.8 或更高版本，以及以下依赖包：

```bash
pip install numpy pandas arrow pyyaml lightgbm scikit-learn meteva matplotlib seaborn scipy joblib
```

对于基于 PyTorch 的演示模型，还需要安装：

```bash
pip install torch
```

## 使用

### 完整流程

```bash
# 1. 将 EC 预报数据提取到站点
python extract_ec.py

# 1b. (仅用于新原始观测数据) 将分站 CSV 转换为 ob{year}.npy
python extract_obs.py

# 2. 将 CSV 文件转换为 npy 数组
python convert_npy.py

# 3. 训练 LightGBM 模型
python lgb_uv.py
# 或
python lgb_wind.py

# 4. 生成预测结果
python lgb_predict.py

# 5. 评估模型性能
python access.py
# 或
python access97.py
# 或
python access97_uv.py
```

### 配置

路径通过 `config/config.yaml` 和本地覆盖文件管理：

- `config/config.yaml` - 共享常量和路径占位符
- `config/config.local.windows.yaml` - Windows 专用路径
- `config/config.local.linux.yaml` - Linux 专用路径

复制本地示例并填写本机的真实路径。这些本地文件会被 Git 忽略。

## 项目结构

```
wind_10m/
├── config/                  # YAML 配置文件
├── extract_ec.py            # 将 EC 预报数据提取到站点
├── extract_obs.py           # 将分站原始观测 CSV 转换为 npy
├── convert_npy.py           # 将 CSV 文件转换为 npy 数组
├── lgb_uv.py                # LightGBM UV 分量训练
├── lgb_wind.py              # LightGBM 风速训练
├── lgb_cv.py                # LightGBM 交叉验证
├── lgb_predict.py           # LightGBM 预测
├── pdfm.py / pdfm_uv.py     # PDF 匹配后处理模型
├── avt.py, cnn.py, dnn.py, emd.py, gn.py, ots.py, transformer.py
│                            # PyTorch 演示模型
├── access.py                # 评估 (旧站集) 
├── access97.py              # 评估 (97 站) 
├── access97_uv.py           # 评估 (97 站，UV 分量) 
├── models/                  # 可复用模型模块
│   └── base_torch.py        # torch 训练共享工具
├── evaluation/              # 可复用绘图模块
│   ├── access97_plots.py
│   └── access97_uv_plots.py
├── src/                     # 共享工具模块
│   └── utils.py
├── README.md
├── README_cn.md
└── AGENTS.md
```

## 里程碑

### 已完成

| 任务 | 描述 | 完成时间 |
|------|------|----------|
| 项目基础建设 | 建立预处理/配置/工具模块结构，统一路径加载 | 2026-04-20 |
| 文档规范化 | 将 `AGENTS.md` 和 `README.md` 重写为英文，创建 `README_cn.md` | 2026-07-21 |
| 编码规范与演示模型正式化 | 正式化 torch 演示模型，全项目应用编码规范 | 2026-07-22 |
| 新原始观测预处理 | 新增 `extract_obs.py`，将分站原始 CSV 直接转换为 `ob{year}.npy` 数组 | 2026-09-14 |

### 进行中

| 任务 | 描述 |
|------|------|
| 新站集实验 | 在更大规模的新原始数据集上重复完整流程，使用全新站集 |

### 待办

| 任务 | 描述 | 优先级 |
|------|------|--------|
| 添加单元测试 | 为核心函数添加测试用例 | P3 |
| 演示模型方案集成 | 重构演示模型以支持统一/分时效/分站点方案 | P2 |

**最后更新**：2026-09-15
**Last Updated**: 2026-09-15

## 许可证

这是一个内部研究项目，不提供公共许可证。

## 作者

- **作者**：yinlb
