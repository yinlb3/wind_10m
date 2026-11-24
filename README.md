<p align="center" style="font-size: 24px; border: none;">
  <strong>English</strong> | <a href="README_cn.md">中文</a>
</p>

# wind_10m

This repository implements ECMWF 10 m wind-speed and wind-direction
post-processing using LightGBM and angle-encoding strategies. It includes
station-observation cleaning, feature alignment, model training, cross-validation,
verification tables, and figures.

The workflow is independent of a fixed station count. The reference experiment
uses a 2019–2024 dataset with 2411 candidate stations and 2187 stations after
quality control.

## Requirements and installation

Python 3.11 matches the verified environment. Dependency versions are pinned in
`requirements.txt`:

```bash
git clone https://github.com/yinlb3/wind_10m.git
cd wind_10m
pip install -r requirements.txt
```

## Input data

### Directory layout

The current data adapter reads per-station NWP NumPy arrays and observation CSVs:

```text
NWP directory/
  data{station_id}.npy   # Reference shape: (4384, 52, 31)

Observation directory/
  {station_id}.csv       # Hourly observations
  2024/{station_id}.csv  # Optional additional observations
```

Observation CSVs must contain:

```text
Station_Id_C, Datetime, PRE_1h, WIN_D_Avg_10mi, WIN_S_Avg_10mi
```

Station IDs are derived from filenames, and NWP and observation files are matched
by ID. The reference lead-time layout, 31-column factor layout, and special
missing-value codes are documented in `docs/guide_cn.md`.

### Local configuration

Create `config/config.2411.local.yaml` with the actual input and output paths:

```yaml
paths:
  data_dir: '/path/to/nwp'
  sta_dir: '/path/to/obs'
  cache_dir: '/path/to/cache'
  pred_dir: '/path/to/predictions'
  result_dir: '/path/to/results'
  station_list: '/path/to/results/qc/stations_ok.txt'

cache:
  mode: none
```

This file is ignored by Git. The evaluation grid is declared by the dataset YAML
(`lead.grid_start`, `lead.grid_end`, `lead.grid_step`) rather than by a run-time
switch, and `cache.mode: none` disables persistent-cache construction. Generic
defaults are in `config/base.yaml`; reference dataset settings are in
`config/datasets/2411.yaml`.

## Reproducing the reference experiment

Run the following commands from the project root, in order:

```bash
# 1. Clean observations and create the QC station list
python qc_stations.py

# 2. Six-fold leave-one-year-out training and prediction
python train.py

# 3. Common-sample verification and grouped statistics
python verify.py

# 4. Mutual-information matrix and figures
python compute_mi.py
python plot.py
```

Each fold includes ECMWF-IFS, PDFM, MLR, LightGBM-A (angle encoding), and
LightGBM-B (vector components).

For small-sample checks, training and verification must use the same `--limit`.
Smoke-check outputs are not final experiment results.

## Output files

Output locations are controlled by `paths` in the local YAML. The reference
configuration uses:

| Content | Directory |
|---|---|
| QC tables and station lists | `docs/results/qc/` |
| Per-model, per-year predictions | `<pred_dir>` |
| Overall and grouped verification tables | `docs/results/tables/` |
| Figures 1–5 | `docs/results/figures/` |
| Optional per-station caches | `<cache_dir>` |

Large outputs (predictions and optional caches) live outside the repository
under a local data root; only the small tables and figures stay in `docs/`.

Metric definitions, common-sample rules, and verified reference values are in
`docs/guide_cn.md`.

## Observation download

Skip this step if observation CSVs are already available. To download hourly
national-station observations from CMADaaS:

```bash
python download_obs.py --national --batch-size 30 \
  /path/to/sta 20160101 20261111
```

For a custom station table:

```bash
python download_obs.py --station-info /path/to/stations.csv \
  --batch-size 30 /path/to/sta 20160101 20261111
```

The downloader uses serial requests and half-year slices. Running the same
command again resumes from `.download_state.json` without repeating completed
station slices. Failures do not remove temporary files; only an explicit
`--restart` resets an incomplete job. Stations without observations receive a
header-only CSV. State and logging details are in `docs/guide_cn.md`.

## Cache configuration

Persistent caches are optional. Estimate their size before construction:

```bash
python cache_size.py --stations 2233 --hours 24
```

| `cache.mode` | Stored arrays | Reference size | Recommended use |
|---|---|---:|---|
| `none` | No files | 0 | One-off runs and minimum disk use |
| `obs` | `dir/spd/pre` | 2.63 GiB | Limited disk; features rebuilt on demand |
| `full` | `x/dir/spd/pre` | 25.38 GiB | Repeated experiments and unified/per-lead schemes |

Reference sizes use the configured station count, samples, evaluation hours,
and features. Use `--stations`, `--samples`, `--hours`, and `--features` for
other datasets.

After selecting `obs` or `full` in the local YAML, run:

```bash
python build_cache.py
```

## Using another dataset

1. Prepare per-station NWP and observation files using the layout above.
2. Copy or edit `config/datasets/2411.yaml`.
3. Review the time axis in `constants`, lead times in `lead`, factor layout in
   `features`, and special codes and cleaning rules in `obs`.
4. Set local paths, then run the cache estimator and QC.
5. Validate a small station sample before running all stations and folds.

Changing station count alone requires no model-code changes. A different raw
column layout or file format requires a `DatasetAdapter` change.

## Extensions

```bash
# Unified, per-lead, per-station, and per-station-per-lead comparison
python benchmark_schemes.py

# 3–240 h wind forecasts at 3-hour intervals
python predict_product.py
```

The extensions have passed small-sample checks but have no final full-run
results. The deployment product interpolates after 72 h; both the extension schemes
and the product still need a full run on the 80-slot grid.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests use temporary synthetic data to cover configuration, command entry
points, data adaptation, caches, and download failure recovery.

## Documentation

- `docs/guide_cn.md`: reference data, reproduction results, caches, downloads,
  and extension parameters.
- `AGENTS.md`: development rules, verification requirements, and safety boundaries.

## Reproducing end to end

Pinned versions are in `requirements.txt`; `environment.yml` builds a matching
conda environment.

```bash
bash reproduce.sh                    # Linux and macOS
```

On Windows, run the same `python -m wind10m ...` commands with the same
interpreter.

## Citation and license

This code is released under the MIT License (see `LICENSE`).

Accompanying manuscript:

> 基于LightGBM和角度编码策略的风速风向短期预报方法

Figures 1–5 are produced by `plot.py` from the verification tables written by
`verify.py`. Citation metadata is in `CITATION.cff`.

- Author: yinlb
- Email: yinlb3@foxmail.com

Last updated: 2026-11-24
