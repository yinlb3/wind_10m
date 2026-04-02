# Wind-10m 项目说明

> 10米风速机器学习预报校正系统
> 最后更新：2026-04-03

---

## 一、项目概述

本项目是一个基于机器学习（主要是LightGBM）的**10米风速预报校正系统**，用于对ECMWF（欧洲中期天气预报中心）数值天气预报产品的10米风速预报进行后处理和偏差校正，以提高预报精度。

### 核心目标
- 读取ECMWF数值天气预报数据（EC4AI格式）
- 提取指定站点（97个国家气象观测站）的预报要素
- 结合观测数据进行机器学习训练
- 生成校正后的风速预报产品
- 评估模型效果并生成可视化图表

---

## 二、项目结构

```
wind_10m/
├── data/                   # 数据目录
│   ├── 97/                # 风速数据（处理后）
│   └── 97-uv/             # UV分量数据（处理后）
├── preprocessing/          # 数据预处理模块
│   ├── extract_ec.py      # ECMWF数据提取（EC4AI → CSV）
│   └── convert_npy.py     # 数据格式转换（CSV → npy）
├── models/                 # 模型训练、预测及后处理脚本
│   ├── LightGBM.py        # LightGBM主训练脚本（UV分量）
│   ├── LightGBM_CV.py     # 交叉验证版本
│   ├── LigthGBM2.py       # 风速量级版本
│   ├── cnn_demo.py        # CNN模型Demo
│   ├── dnn_demo.py        # DNN模型Demo
│   ├── transformer_demo.py # Transformer模型Demo
│   ├── avt.py             # 平均值调整技术(AVT)
│   ├── emd.py             # 经验模态分解(EMD)
│   ├── gn.py              # 梯度提升网络相关
│   ├── ots.py             # 最优阈值选择
│   ├── pdfm.py            # 概率密度函数匹配(PDF)
│   └── pdfm_uv.py         # UV分量的PDF匹配
├── src/                    # 通用工具模块
│   └── utils.py           # 常用工具函数（时间格式化等）
├── evaluation/             # 模型评估模块
│   ├── access.py          # 通用评估工具（散点图、时间格式化）
│   ├── access97.py        # 97站点评估（风速、风向评分）
│   └── access97_uv.py     # UV分量评估（U/V风分量详细评分）
└── lgb_predict.py         # LightGBM预测脚本
```

---

## 三、核心模块说明

### 3.1 数据处理流程

| 步骤 | 脚本 | 功能描述 |
|------|------|----------|
| 1 | `preprocessing/extract_ec.py` | 从EC4AI二进制文件提取30个气象要素到97个站点 |
| 2 | `preprocessing/convert_npy.py` | 读取国家站观测数据（10分钟平均风速等8个要素） |


**提取的EC要素**（30个）：
- 地面：2T(2米温度), MSL(海平面气压), 2D(2米露点), 10U(10米U风), 10V(10米V风)
- 各层（1000/950/925/900/850 hPa）：T(温度), GH(位势高度), R(相对湿度), U/V(风分量)

### 3.2 模型训练

**主脚本**：`models/LightGBM.py`

- **模型**：LightGBM回归器
- **训练策略**：6折交叉验证（留一年做验证）
- **训练年份**：2017-2022年
- **输入特征**：30个EC要素
- **输出目标**：10分钟平均风速（或U/V分量）
- **时间窗**：预报时效12-36小时，逐小时插值

**训练模式**（4种）：
1. 统一模型（所有时次、站点共享一个模型）
2. 分预报时效模型（每个预报时效一个模型）
3. 分站点模型（每个站点一个模型）- 主用
4. 分时效+站点模型（每个时效-站点组合一个模型）

### 3.3 模型评估

**评估脚本**：`evaluation/access.py`, `evaluation/access97.py`, `evaluation/access97_uv.py`

**评估维度**：
- 分时次（12-36小时）
- 分月份（1-12月）
- 分量级（0.3m/s, 1.6m/s, 3.4m/s, 5.5m/s, 8.0m/s, 10.8m/s, 13.9m/s, 17.2m/s）
- 分站点（97个站点空间分布）

**评估指标**：
- R（相关系数）
- ME（平均误差）
- MAE（平均绝对误差）
- RMSE（均方根误差）
- MRE（平均相对误差）
- TS（Threat Score）
- ETS（Equitable Threat Score）

### 3.4 后处理技术（utils/）

| 模块 | 功能 | 算法说明 |
|------|------|----------|
| `avt.py` | 平均值调整 | 基于观测和预报的均值方差调整 |
| `pdfm.py` | 概率密度匹配 | 匹配观测与预报的累积概率分布 |
| `emd.py` | 经验模态分解 | 信号分解与去噪 |
| `ots.py` | 最优阈值 | 基于TS评分的最优分类阈值 |

---

## 四、关键依赖

```
- Python 3.8+
- numpy
- pandas
- arrow（时间处理）
- lightgbm（主模型）
- scikit-learn（MLR等基线模型）
- meteva（气象检验评估库）
- matplotlib（绘图）
- seaborn（统计可视化）
- scipy（科学计算）
- joblib（模型序列化）
```

---

## 五、数据格式

### 输入数据
- **EC预报**：`.AI.bin` 二进制格点文件（0.125°/0.25°分辨率）
- **观测数据**：CSV格式，包含风向风速等要素
- **站点信息**：CSV格式（台站号、经度、纬度、海拔）

### 中间数据（.npy格式）
```
nwp{year}.npy  - ECMWF提取的站点预报数据，shape: (样本数, 85时效, 97站点, 30要素)
ob{year}.npy   - 观测数据，shape: (样本数, 85时效, 97站点, 8要素)
```

### 输出数据
```
lgb_{option}_{year}.npy    - LightGBM预测结果
mlr_{option}_{year}.npy    - MLR基线模型结果
avt_{year}.npy             - AVT后处理结果
```

---

## 六、编码规范

### 6.1 文件头部模板
所有Python文件遵循统一头部格式：

```python
#!user/bin.python3

"""
Founded in YYYY-MM-DD
Modified in YYYY-MM-DD
@author: yinlb
"""
```

### 6.2 程序入口模板
```python
if __name__ == '__main__':
    print('The program "<filename>.py" is beginning.')
    start = arrow.now()
    
    # 主逻辑
    
    end = arrow.now()
    running_time = (end - start).total_seconds()
    print('The program "<filename>.py" runs out in {:s}.'.format(format_time(running_time)))
```

### 6.3 命名规范
- 函数/变量：小写+下划线（`snake_case`）
- 常量：全大写（`ELEMENTS`, `THRES`）
- 类名：大驼峰（`Acc`, `PDF`, `AVT`）
- 中文注释，文档字符串使用中文

---

## 七、运行说明

### 7.1 完整流程

```bash
# 1. 提取EC数据到站点
python preprocessing/extract_ec.py

# 2. 转换数据格式（CSV → npy）
python preprocessing/convert_npy.py

# 3. 训练LightGBM模型
python models/LightGBM.py

# 4. 生成预测
python lgb_predict.py

# 5. 评估模型效果
python evaluation/access.py
# 或 python evaluation/access97.py
# 或 python evaluation/access97_uv.py
```

### 7.2 路径配置

默认路径（可在各脚本 `if __name__ == '__main__'` 中修改）：
- EC数据：`\\10.110.173.91\sqxt\EC4AI`
- 观测数据：`D:\data\国家站`
- 输出数据：`D:\data\wind`, `D:\model\wind2`
- 站点信息：`D:\Project\wind\国家气象观测站.csv`

---

## 八、模型版本说明

| 版本 | 说明 | 路径 |
|------|------|------|
| wind | 早期版本（单一时次） | D:\data\wind |
| wind2 | UV分量版本 | D:\model\wind2 |
| wind5 | 风速量级版本 | D:\model\wind5 |
| 97 | 97站点验证 | D:\Project\wind\97 |
| 97-uv | UV分量验证 | D:\Project\wind\97-uv |

---

## 九、注意事项

1. **编码问题**：CSV文件使用 `gb2312` 或 `gbk` 编码读取
2. **缺失值处理**：使用 `np.nan` 标记，训练时自动过滤
3. **时间处理**：使用 `arrow` 库，UTC+8时区
4. **内存管理**：大数据使用 `.npy` 格式分块处理
5. **并行训练**：LightGBM使用 `n_jobs=20` 加速

---

## 十、项目里程碑

### ✅ 已完成

| 任务 | 描述 | 完成时间 |
|------|------|----------|
| 数据预处理模块化 | 创建 `preprocessing/` 目录，移动 `ec_station.py` → `extract_ec.py`，`data.py` → `convert_npy.py` | 2026-04-02 |
| 项目结构优化 | 抽象 `format_time` 到 `src/utils.py`，原 `utils/` 移至 `models/`，统一使用中文 Docstring | 2026-04-03 |
| 删除冗余文件 | 移除 `merge_data.py`（非本项目文件）和 `merge_sta.py`（大赛专用脚本） | 2026-04-02 |
| 修复索引越界 | `convert_npy.py` 添加站点存在性检查，避免 `ValueError` | 2026-04-02 |
| 同步 ELEMENTS | `extract_ec.py` 和 `convert_npy.py` 要素列表保持一致 | 2026-04-02 |
| 修复变量未定义 | `extract_ec.py` 添加 `file` 变量定义 | 2026-04-02 |
| 评估模块整理 | 创建 `evaluation/` 目录，移动 `access*.py` 评估脚本 | 2026-04-03 |

### 🔄 进行中

| 任务 | 描述 | 备注 |
|------|------|------|
| 代码审查与整理 | 检查各模块代码质量，确保数据流正确 | 重点检查时效匹配逻辑 |
| 文档同步更新 | 随代码变更同步更新 AGENTS.md | - |
| MCP 问题反馈 | 向 Kimi Code 官方反馈 MCP 无法加载问题 | Issue #138 已提交，等待修复 |

### 📋 待办事项

| 任务 | 描述 | 优先级 |
|------|------|--------|
| 模型训练脚本检查 | 检查 `models/` 目录各脚本是否正确引用预处理后的数据路径 | P1 |
| 评估脚本检查 | 检查 `evaluation/access*.py` 是否能正确读取模型输出 | P1 |
| 路径配置统一 | 考虑将硬编码路径提取到配置文件 | P2 |
| 代码注释完善 | 补充关键算法的注释说明 | P3 |
| 添加单元测试 | 为核心函数添加测试用例 | P3 |

---

## 十一、作者与维护

- **作者**：yinlb
- **创建时间**：2023-07
- **最后修改**：2026-04-03
- **项目路径**：`D:\Project\wind_10m`
