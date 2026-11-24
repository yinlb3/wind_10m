---
name: wind-10m
description: Configurable ECMWF 10 m wind forecast correction workflow.
---

# Wind-10m Agent Guide

> Maintain a reusable workflow. The station dataset is a validated
> reference example, not the project's purpose or a station-count limit.

## 1. Scope and status

Wind-10m cleans station observations, aligns NWP features, trains correction
models, and writes predictions, verification tables, and figures. The reference
implementation includes LightGBM angle/vector variants and ECMWF-IFS, PDFM,
and MLR baselines.

- Reference data: 2019–2024, 2233 usable stations, 2187 after QC.
- Completed: six-fold predictions, common-sample tables 4/5, figures 1–5.
- Pending: old-result comparison for deeper refactoring, full four-scheme
  results, and full deployment products.

## 2. Current structure

```text
wind10m/
  cli.py                    Unified python -m wind10m dispatcher
  config.py                 Generic + dataset + local YAML loading
  cache.py                  Cache constants and pure size estimates
  memory.py                 Available-RAM probe and station-block sizing
  dataset/
    base.py                 DatasetAdapter interface
    npy_csv.py              Current per-station NPY/CSV adapter
config/
  base.yaml                 Generic defaults; cache.mode defaults to none
  datasets/2411.yaml        Reference data dimensions and paths
  config.2411.local.yaml    Optional private reference paths; ignored
src/workflow.py             Reference numerical data implementation
src/window.py               3–240 h interpolation and observation alignment
cache_size.py               Compatible cache estimator entry point
build_cache.py         Optional per-station cache builder
qc_stations.py         Observation QC and required station list
train.py               Six-fold, five-model reference training
verify.py              Common-sample and grouped verification
compute_mi.py          Mutual-information matrix
plot.py                Result figures
benchmark_schemes.py        Four-scheme comparison, tables 6/7
predict_product.py            deployment wind product
download_obs.py             Optional standalone observation downloader
tests/                      Synthetic configuration/CLI/adapter/cache tests
docs/guide_cn.md             Reference data, results, extensions, refactor status
LICENSE                     MIT license
CITATION.cff                Citation metadata for the accompanying paper
environment.yml             Conda environment; pins live in requirements.txt
reproduce.sh                End-to-end qc, train, verify, mi and plot commands
```

The generic shell delegates numerical operations to the reference code. Do not
change training, schemes, or evaluation internals until old-result comparisons
are available. A new raw format requires another `DatasetAdapter`.

## 3. Configuration and dataset replacement

Configuration merges `config/base.yaml`, the selected dataset YAML, and an
optional local YAML. `WIND10M_CONFIG` and `WIND10M_LOCAL` are set by the unified
CLI and honored by compatibility entry points.

The current adapter expects:

```text
data_dir/data{station_id}.npy   (n_samples, n_leads, n_columns)
sta_dir/{station_id}.csv        Hourly observations
sta_dir/2024/{station_id}.csv   Optional reference-data split
```

For a replacement dataset:

1. Copy and adjust a dataset YAML, including time, lead, feature, obs, and paths.
2. Confirm raw factor indices match the adapter or implement a new adapter.
3. Pass `--config` and optionally `--local` before the workflow command.
4. Estimate cache storage, then run QC to create a new station list.
5. Run the synthetic suite and a small real-data smoke check before full runs.

Station IDs come from filenames. Available stations are the NWP/observation
intersection, filtered by the configured QC station list.

## 4. Commands

### Reference experiment

The evaluation grid is declared by the dataset YAML (`lead.grid_start`,
`lead.grid_end`, `lead.grid_step`); the reference dataset spans 3–240 h at 3 h
steps. Changing the grid is a dataset change, not a run-time switch.

```bash
python -m wind10m --config config/datasets/2411.yaml \
  estimate-cache --stations 2233
python -m wind10m --config config/datasets/2411.yaml qc
# Optional: set cache.mode to obs or full first
python -m wind10m --config config/datasets/2411.yaml cache
python -m wind10m --config config/datasets/2411.yaml train
python -m wind10m --config config/datasets/2411.yaml verify
python -m wind10m --config config/datasets/2411.yaml mi
python -m wind10m --config config/datasets/2411.yaml plot
```

Omitting `--config` selects the reference dataset. Existing root scripts remain
compatible and use the same loader. Training and verification require
`paths.station_list` and matching `--limit` values for smoke checks.

### Cache modes

- `none`: default; build no files and preprocess in memory.
- `obs`: store `dir/spd/pre`; rebuild features and targets on load.
- `full`: store `x/dir/spd/pre`; recommended for repeated experiments and
  unified/per-lead benchmarks.

Both persistent formats reconstruct `y/uv` and accept legacy full caches.
`estimate-cache` supports station, sample, hour, and feature overrides.

### Extensions

The four-scheme comparison and the deployment product use the same dataset grid and
write into the configured prediction and result directories.

```bash
python -m wind10m --config config/datasets/2411.yaml benchmark
python -m wind10m --config config/datasets/2411.yaml product
```

No limiting arguments means all configured stations and folds. `--limit` and
`--subset` sample stations at random with a fixed seed (42; override with
`--seed`), so a small run reproduces exactly and avoids the spatial bias of a
head slice. Any `--limit`, `--subset`, or incomplete `--years` run is still a
smoke check. The deployment product uses interpolation after 72 h and lacks
independent later-lead verification.

## 5. Verification

Run after Python changes:

```bash
python -m py_compile <file>.py
python -m unittest discover -s tests -v
```

The standard-library suite uses temporary synthetic inputs to test real config,
CLI, adapter, QC, cache, and download failure/recovery paths. It does not prove
equivalence with old full experiment predictions.

When old-result data are available:

1. Compare representative raw stations and all array keys exactly.
2. Run a small one-fold, five-model comparison for every cache mode.
3. Run one full-station fold and compare predictions and verification tables.
4. Keep unexpected relative metric differences within 0.1% and investigate any
  change in common-sample counts.
5. Only then migrate numerical internals or retire compatibility entry points.

Never run a full pipeline that overwrites large outputs without explicit user
confirmation.

## 6. Code style

- Python >=3.11 is required by the minimum dependency versions.
- English comments, docstrings, output, and exception messages.
- External data columns and chart labels may remain Chinese.
- Use `pathlib`, YAML configuration, Google-style docstrings, type annotations,
  and lines within 80 characters.
- Persist data, caches, labels, and predictions as float32. LightGBM Sequence
  rows may be float64 only at the library interface.
- Never hardcode private paths.
- Update `Modified` dates in edited Python file headers.
- Keep both READMEs, this guide, and `docs/guide_cn.md` synchronized.

## 7. Git and safety boundaries

### Always

- Run syntax and focused functional checks before reporting completion.
- Keep generated data, caches, predictions, results, and local config out of Git.
- Review `git status` and remove only temporary files created by the current task.

### Ask first

- Delete or rename existing model outputs.
- Add runtime dependencies or change numerical thresholds/constants.
- Run full experiments with large output or runtime requirements.

### Never

- Commit or push without explicit user approval.
- Commit ignored local configuration, credentials, or private paths.
- Treat smoke outputs or estimated metrics as final results.
- Reintroduce retired legacy workflows into user-facing documentation.

## 8. Completed and pending work

Completed on this computer on 2026-11-24:

- Declared the evaluation grid in the dataset YAML instead of a run-time mode
  switch; predictions and results no longer fork into per-mode directories.
- Reworked verification to memory-map predictions and score adaptive station
  blocks; the whole-fold arrays needed roughly 46 GB at 80 hours.
- Stored a cache descriptor in every NPZ and refused incompatible caches with a
  single rebuild notice.
- Added `--years` to verification and fixed `--limit`, which previously could
  not slice full-station prediction files.
- Every output now targets the grid declared by the dataset; no per-mode
  output directories remain.
- Added the MIT license, citation metadata, pinned requirements, a conda
  environment and an end-to-end `reproduce.sh`.

Completed on this computer on 2026-11-23:

- Added optional `none/obs/full` caches and lossless target reconstruction.
- Added raw-observation QC and dataset-specific cache size estimation.
- Added layered generic/dataset/local configuration and the unified CLI.
- Added the adapter interface and NPY/CSV implementation without changing
  numerical algorithms.
- Added resumable serial downloads with per-station completion, failure
  preservation, exact retry context, bounded logs, and empty-station CSVs.
- Removed the retired station-specific entry points, evaluation helpers and
  legacy configuration files from the supported codebase.
- Added nine synthetic regression tests and retained current entry-point
  compatibility.
- Merged reference and extension documentation into `docs/guide_cn.md`.

Pending:

- `download_obs.py` imports `nmc_met_io` at module level, so importing it (or
  the download tests) fails when `~/.nmcdev/config.ini` is missing; move the
  import into the request path and document the credentials file.
- Small real CMADaaS requests and targeted replay of the previous empty slice.
- Before public release: complete old-result and one-fold validation, then audit Git history for private content.
- Migration of training, schemes, and evaluation internals after equivalence.
- Additional adapters for other raw layouts, including yearly arrays.
- Full four-scheme and deployment runs with independent later-lead verification.

**Last Updated**: 2026-11-24
