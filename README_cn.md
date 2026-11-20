# Wind-10m

> 基于 ECMWF 数值天气预报产品的 10 米风速机器学习预报校正系统

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3%2B-green)]()

基于 LightGBM 的 ECMWF 10 米风速预报后处理与偏差校正系统。

## 目录

- [安装](#安装)
- [使用](#使用)
- [项目结构](#项目结构)
- [许可证](#许可证)
- [里程碑](#里程碑)
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

# 1a.（可选）下载 paths.station_csv 中站点的观测数据
# 日期格式为 YYYYMMDD，结束日期不包含在下载范围内
python download_obs.py 20170101 20230101
# 或使用 meteva 内置国家站列表
python download_obs.py --national 20170101 20230101
# 可选：均衡分批，每批最多 100 站（默认全站）
python download_obs.py --national --batch-size 100 20170101 20230101

# 1b.（仅用于新原始观测数据）将分站 CSV 转换为 ob{year}.npy
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
# 或
python access97_sc.py
```

### 配置

路径通过 `config/config.yaml` 和本地覆盖文件管理：

- `config/config.yaml` - 共享常量和路径占位符
- `config/config.local.windows.yaml` - Windows 专用路径
- `config/config.local.linux.yaml` - Linux 专用路径

复制本地示例并填写本机的真实路径。这些本地文件会被 Git 忽略。

若 `meb.read_station()` 报 micaps3 格式错误（pandas >= 2），修复方法参见 <https://github.com/nmcdev/meteva/issues/193>。

## 项目结构

```
wind_10m/
├── config/                  # YAML 配置文件
├── download_obs.py          # 从 CMADaaS 下载分站观测 CSV
├── extract_ec.py            # 将 EC 预报数据提取到站点
├── extract_obs.py           # 将分站原始观测 CSV 转换为 npy
├── convert_npy.py           # 将 CSV 文件转换为 npy 数组
├── lgb_uv.py                # LightGBM UV 分量训练
├── lgb_wind.py              # LightGBM 风速训练
├── lgb_cv.py                # LightGBM 交叉验证
├── lgb_sc.py                # LightGBM SC 编码训练变体
├── lgb_predict.py           # LightGBM 预测
├── pdfm.py / pdfm_uv.py     # PDF 匹配后处理模型
├── avt.py, cnn.py, dnn.py, emd.py, gn.py, ots.py, transformer.py
│                            # PyTorch 演示模型
├── access.py                # 评估 (旧站集)
├── access97.py              # 评估 (97 站)
├── access97_uv.py           # 评估 (97 站，UV 分量)
├── access97_sc.py           # 评估 (97 站，SC 编码预报)
├── rename_vars.py           # 重命名 Python 源码中的变量名
├── models/                  # 可复用模型模块
│   └── base_torch.py        # torch 训练共享工具
├── evaluation/              # 可复用绘图模块
│   ├── access97_plots.py
│   ├── access97_sc_plots.py
│   └── access97_uv_plots.py
├── src/                     # 共享工具模块
│   ├── metrics.py           # 共享评估指标 (Acc 类)
│   └── utils.py
├── README.md
├── README_cn.md
└── AGENTS.md
```

## 2411 站实验复现

2411 站数据已是站点级 numpy 数组，不再需要 `extract_ec.py` / `convert_npy.py` / `extract_obs.py`。

```bash
# 0. 环境：C:\ProgramData\miniconda3\envs\meteva\python.exe（Python 3.11）
pip install -r requirements.txt

# 1. 为全部 2233 个可用站建缓存（约 6 min，约 30 GB）
python build_cache_2411.py

# 2. 站点质控：保留 2187 站并生成 paths.station_list 清单
python qc_stations_2411.py

# 3. 训练 5 个模型，6 折留一年（约 5 h）
python train_2411.py

# 4. 输出检验表到 docs/results/tables
python verify_2411.py

# 5. 互信息矩阵与出图
python compute_mi_2411.py
python plot_2411.py
```

本机路径放 `config/config.2411.local.yaml`（Git 忽略），共享配置在
`config/config.2411.yaml`。训练和检验会在 QC 站点清单缺失时明确停止。
步骤清单与验收标准见 `docs/repro-2411_cn.md`。

## 许可证

这是一个内部研究项目，不提供公共许可证。

## 里程碑

### 已完成

| 任务 | 描述 | 完成时间 |
|------|------|----------|
| 项目基础建设 | 建立预处理/配置/工具模块结构，统一路径加载 | 2026-04-20 |
| 文档规范化 | 将 `AGENTS.md` 和 `README.md` 重写为英文，创建 `README_cn.md` | 2026-07-21 |
| 编码规范与演示模型正式化 | 正式化 torch 演示模型，全项目应用编码规范 | 2026-07-22 |
| 新原始观测预处理 | 新增 `extract_obs.py`，将分站原始 CSV 直接转换为 `ob{year}.npy` 数组 | 2026-09-14 |
| 文档清理 | 解决 `AGENTS.md`、`README.md`、`README_cn.md` 中残留的 Git 冲突标记，并使文件结构与命令与实际布局一致 | 2026-09-17 |
| 2411 站主实验 | 完成 2187 站质控、六折预测、共同样本表图和论文修订 | 2026-11-06 |
| 扩展与复现加固 | 修复扩展掩膜/汇总和产品输出，补齐 QC 步骤并同步文档 | 2026-11-19 |
| CMADaaS 下载器增强 | 新增可配置均衡分批、半年请求、60 要素下载和按站点原子写入 CSV | 2026-11-20 |

### 进行中

| 任务 | 描述 |
|------|------|
| 扩展全量运行 | 四方案对比和十天产品；代码与合成检查通过，全量运行尚未安排 |

### 待办

| 任务 | 描述 | 优先级 |
|------|------|--------|
| 添加单元测试 | 为核心函数添加测试用例 | P3 |
| 演示模型方案集成 | 重构演示模型以支持统一/分时效/分站点方案 | P2 |

## 作者

- **作者**：yinlb<yinlb3@foxmail.com>

**最后更新**：2026-11-21

**[English](README.md)**
