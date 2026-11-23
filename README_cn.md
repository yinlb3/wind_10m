<p align="center" style="font-size: 24px; border: none;">
  <a href="README.md">English</a> | <strong>中文</strong>
</p>

# wind_10m

本仓库实现基于 LightGBM 和角度编码策略的 ECMWF 10 米风速、风向预报后处理，
包括站点观测清洗、特征对齐、模型训练、交叉验证、检验表和图件生成。

流程与具体数据集解耦。当前以 2019–2024 年的 2411 站数据集作为参考样例，
质控后使用 2187 站完成六折实验。

## 环境要求与安装

最低依赖版本要求 Python ≥ 3.11，所有依赖均在 `requirements.txt` 中使用
大于等于版本约束：

```bash
git clone https://github.com/yinlb3/wind_10m.git
cd wind_10m
pip install -r requirements.txt
```

## 输入数据

### 目录结构

当前数据适配器读取逐站 NWP NumPy 数组和逐站观测 CSV：

```text
NWP 目录/
  data站号.npy          # 参考形状：(4384, 52, 31)

观测目录/
  站号.csv              # 逐小时观测
  2024/站号.csv         # 可选的额外观测文件
```

观测 CSV 必须包含以下列：

```text
Station_Id_C, Datetime, PRE_1h, WIN_D_Avg_10mi, WIN_S_Avg_10mi
```

站号由文件名确定，NWP 和观测文件按站号匹配。参考数据集的 52 个原始时效、
31 列因子布局及特殊缺测码见 `docs/2411_cn.md`。

### 本机配置

新建 `config/config.2411.local.yaml`，填写实际输入和输出目录：

```yaml
paths:
  data_dir: '/path/to/nwp'
  sta_dir: '/path/to/obs'
  cache_dir: '/path/to/cache'
  pred_dir: '/path/to/predictions'
  result_dir: '/path/to/results'
  station_list: '/path/to/results/qc/stations_ok.txt'

lead:
  mode: 24

cache:
  mode: none
```

该文件已被 Git 忽略。`lead.mode: 24` 对应已验证的 13–36 h 主实验窗口；
`cache.mode: none` 表示不建立持久缓存。通用默认参数位于
`config/base.yaml`，参考数据集参数位于 `config/datasets/2411.yaml`。

## 复现参考实验

以下命令均在项目根目录执行，并按顺序运行：

```bash
# 1. 观测清洗与站点质控，生成 stations_ok.txt
python qc_stations_2411.py

# 2. 六折留一年训练与预测
python train_2411.py

# 3. 共同样本检验及分组统计
python verify_2411.py

# 4. 互信息矩阵与图件
python compute_mi_2411.py
python plot_2411.py
```

每折包含 ECMWF-IFS、PDFM、MLR、LightGBM-A（角度编码）和
LightGBM-B（矢量分解）五个模型。

小样本检查时，训练和检验必须使用相同的 `--limit`；冒烟输出不属于正式
实验结果。

## 输出文件

输出位置由本机 YAML 中的 `paths` 决定。参考配置使用：

| 内容 | 目录 |
|---|---|
| QC 表格及站点清单 | `docs/results/qc/` |
| 逐模型、逐年份预测 | `artifacts/2411/pred/` |
| 总体与分组检验表 | `docs/results/tables/` |
| 图 1–5 | `docs/results/figures/` |
| 可选逐站缓存 | `artifacts/2411/cache/` |

参考实验的指标定义、共同样本口径和已核验数值见 `docs/2411_cn.md`。

## 观测数据下载

已有观测 CSV 时无需执行本步骤。使用 CMADaaS 下载国家站逐小时观测：

```bash
python download_obs.py --national --batch-size 30 \
  /path/to/sta 20160101 20261111
```

自定义站表使用：

```bash
python download_obs.py --station-info /path/to/stations.csv \
  --batch-size 30 /path/to/sta 20160101 20261111
```

下载器采用串行请求和半年切片。再次执行相同命令可从
`.download_state.json` 恢复，不重复已完成的时间片。失败不会删除临时文件；
只有显式指定 `--restart` 才重置未完成任务。整段无观测的站输出表头空 CSV。
运行状态和日志说明见 `docs/2411_cn.md`。

## 缓存配置

持久缓存为可选项。建立前应先估算容量：

```bash
python cache_size.py --stations 2233 --hours 24
```

| `cache.mode` | 保存内容 | 参考大小 | 适用场景 |
|---|---|---:|---|
| `none` | 不写文件 | 0 | 一次性实验、最小磁盘占用 |
| `obs` | `dir/spd/pre` | 2.63 GiB | 磁盘有限，特征按需重算 |
| `full` | `x/dir/spd/pre` | 25.38 GiB | 反复实验、统一和分时效方案 |

参考大小基于 2233 站、4384 样本、24 时次和 26 个因子。其他数据体量可用
`--stations`、`--samples`、`--hours` 和 `--features` 估算。

在本机 YAML 中选择 `obs` 或 `full` 后执行：

```bash
python build_cache_2411.py
```

## 使用其他数据集

1. 按上述目录结构准备逐站 NWP 和观测文件。
2. 复制或修改 `config/datasets/2411.yaml`。
3. 核对 `constants` 中的时间轴、`lead` 中的时效、`features` 中的因子布局，
   以及 `obs` 中的特殊码和清洗规则。
4. 设置本机路径，运行容量估算和 QC。
5. 小样本验证通过后再运行全站、全折实验。

仅改变站点数量不需要修改模型代码。原始列布局或文件格式不同，需要调整
`DatasetAdapter`。

## 扩展功能

```bash
# 统一、分时效、分站、分站分时效四方案对比
python benchmark_schemes.py

# 3–240 h、逐 3 h 风预报产品
python predict_10day.py
```

扩展代码已通过小样本检查，但尚未形成全量正式结果。十天产品在 72 h 后
使用线性插值；当前仅 13–36 h 经过主实验检验，更晚时效需要独立观测验证。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试使用临时合成数据，覆盖配置、命令入口、数据适配、缓存以及下载失败恢复。

## 文档

- `docs/2411_cn.md`：参考数据、复现结果、缓存、下载和扩展参数。
- `AGENTS.md`：开发规范、验证要求和安全边界。

## 许可证与联系方式

本项目为内部研究代码，未提供公共许可证。

- 作者：yinlb
- 邮箱：yinlb3@foxmail.com

最后更新：2026-11-23
