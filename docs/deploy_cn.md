# 2411 站扩展交付物

扩展脚本（`predict_10day.py` / `benchmark_schemes.py`，均在项目根目录）与
主实验（`build_cache_2411.py` / `train_2411.py` /
`verify_2411.py`）隔离：十天产品写入 `artifacts/2411/deploy`，四方案结果
使用独立的表 6/7 文件名，不覆盖主实验预测和表 4/5。

> 当前状态：代码已完成冒烟验证，尚未全量运行；其结果不作为论文结论。
> 主实验的 2187 站、六折表图不受影响。

## 1. 十天逐 3 h 风预报产品

入口：`predict_10day.py`（项目根目录）

| 项 | 说明 |
|---|---|
| 训练窗口 | 原始 52 个时效（3–240 h）全部参与训练 |
| 推理窗口 | 80 个 3 h 时效（3, 6, …, 240 h） |
| 时效扩展 | 3–72 h 为原始值；72 h 之后按线性内插补出 3 h 中间点 |
| 特征 | 26 个，与主实验同源（降水同样换算为 mm/h） |
| 输出 | `artifacts/2411/deploy/{站号}.npz`，shape (4384, 80) |
| 数组类型 | `dir` / `dir_b` / `spd` / `spd_b` 全部 float32 |
| 磁盘 | A 约 4.2 MB/站；A+B 约 8.4 MB/站，2187 站双模型约 18 GB |
| 断点 | 仅当 npz 包含本次要求的全部模型数组时才跳过 |

方向使用 float32 是为完整保留静风标记 `999017`（float16 会溢出为 `inf`）；
风速同样用 float32，与项目“数据一律 float32”的规范一致。
`LightGBM-A` 固定写 `dir` / `spd`，`LightGBM-B` 固定写
`dir_b` / `spd_b`，与命令行模型顺序无关。先跑 A、再跑 A+B 时，已有
A-only 文件会补齐 B，不会被错误跳过。

```bash
python predict_10day.py                          # 全量（默认，2187 站）
python predict_10day.py --limit 20               # 仅前 20 站（冒烟，结果丢弃）
python predict_10day.py --models LightGBM-A,LightGBM-B
```

**必须知道的三个限制**

1. **72 h 之后的 3 h 分辨率是插值出来的，不是新增预报信息**。论文与业务
   引用时应标明“后段为线性内插”。
2. 主实验只在 13–36 h 上做过检验。后段（>36 h）**没有经过独立观测检验**，
   不宜直接宣称后段精度。应先用同期观测对齐 13–36 h 段的独立 RMSE。
3. **训练按 (样本, 时次) 逐行筛选**，某个时次缺测只丢该行；推理仍要求
   80 个时次的特征齐全，缺一个则整样本输出 NaN，不做前向填充。

## 2. 表 6 / 表 7：四组建模方案对比

入口：`benchmark_schemes.py`（项目根目录；已实现，尚待全量实跑）

| 方案 | 定义 |
|---|---|
| 一 统一 | 所有站、所有时效合并训练 |
| 二 分时效 | 每个时效一个模型，跨站合并训练 |
| 三 分站 | 每站一个模型（主实验方案） |
| 四 分站分时效 | 每站每时效一个模型 |

**公平性口径**

1. 四个方案使用相同的训练年份、检验年份、完整特征掩膜和完整目标掩膜；
   静风编码不会作为普通数值进入任一目标。
2. 不传参数即全站、六折、四方案全跑。`--subset N` 只在需要缩小时使用，
   且对本次运行的**所有方案**生效；默认 0 表示全站。方案四全站计算量极大。
3. 总表通过 `ScoreAcc.merge()` 合并各折平方误差、计数和混淆矩阵，不使用
   逐折 RMSE 的简单平均；输出同时保留 `n_stations` 和实测 `minutes`。
4. 四方案只比较 LightGBM-A 的建模组织方式，不包含另行实现的 MLR 基线。

**数据读取与并行**

方案一/二可选两种读取方式，由 `model.stream` 决定：

- `auto`（默认）：读取可用内存，需求低于可用内存的 1/1.3 时预加载，否则流式；
- `true`：用 `lightgbm.Sequence` 流式读取，32 GB 机器可用，但统一方案因流式
  回调而偏慢；
- `false`：把全部训练行预加载为内存矩阵，统一方案约需 60 GB、分时效约需
  73 GB，四方案读取口径一致、用时可直接比较。

运行前会打印 `Preload: need=... available=... stream=...`，并写入
`table6_meta.txt`。并行预算为 `inner × outer ≈ parallel.n_total`：
统一内部并行、分时效 2 进程、分站 4 进程、分站分时效 N 进程（按站分块）。

**输出**

- `docs/results/tables/table6_by_fold.csv|txt`：逐折总指标与用时。
- `docs/results/tables/table6_schemes.csv|txt`：六折合并指标、站数与总用时。
- `docs/results/tables/table7_by_lead.csv|txt`：逐折、逐时效指标。
- `docs/results/tables/table7_overall.csv|txt`：六折合并的逐时效指标。

```bash
python benchmark_schemes.py                    # 全量：四方案、2187 站、六折
python benchmark_schemes.py --cores 1 --limit 2 --years 2019   # 单核实测
python benchmark_schemes.py --no-stream --limit 3 --years 2019 # 预加载路径
```

冒烟只用于确认各方案样本口径、输出文件和内存占用，其结果一律不入表。