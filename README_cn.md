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
python preprocessing/extract_ec.py

# 2. 将 CSV 文件转换为 npy 数组
python preprocessing/convert_npy.py

# 3. 训练 LightGBM 模型
python models/LightGBM_uv.py
# 或
python models/LightGBM_wind.py

# 4. 生成预测结果
python models/lgb_predict.py

# 5. 评估模型性能
python evaluation/access.py
# 或
python evaluation/access97.py
# 或
python evaluation/access97_uv.py
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
├── preprocessing/           # EC4AI 提取和 CSV → npy 转换
│   ├── extract_ec.py
│   └── convert_npy.py
├── models/                  # 模型训练、预测和后处理
│   ├── LightGBM_uv.py
│   ├── LightGBM_wind.py
│   ├── LightGBM_CV.py
│   ├── lgb_predict.py
│   ├── avt.py
│   ├── emd.py
│   ├── gn.py
│   ├── ots.py
│   ├── pdfm.py
│   └── pdfm_uv.py
├── evaluation/              # 模型评估和可视化
│   ├── access.py
│   ├── access97.py
│   └── access97_uv.py
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

### 进行中

| 任务 | 描述 | 备注 |
|------|------|------|
| 代码审查与清理 | 检查各模块代码质量，确保数据流正确 | 重点关注时效匹配逻辑 |

### 待办

| 任务 | 描述 | 优先级 |
|------|------|--------|
| 添加单元测试 | 为核心函数添加测试用例 | P3 |
| 演示模型方案集成 | 重构演示模型以支持统一/分时效/分站点方案 | P2 |

**最后更新**：2026-07-22
**Last Updated**: 2026-07-22

## 许可证

这是一个内部研究项目，不提供公共许可证。

## 作者

- **作者**：yinlb
