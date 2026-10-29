# Wind-10m

> 10m wind speed machine learning forecast correction system for ECMWF NWP products.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3%2B-green)]()

A post-processing and bias-correction system for ECMWF 10m wind speed forecasts using LightGBM.

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)
- [Milestones](#milestones)
- [Author](#author)

## Install

Requires Python 3.8 or later and the following packages:

```bash
pip install numpy pandas arrow pyyaml lightgbm scikit-learn meteva matplotlib seaborn scipy joblib
```

For the PyTorch-based demo models, also install:

```bash
pip install torch
```

## Usage

### Full pipeline

```bash
# 1. Extract EC forecast data to stations
python extract_ec.py

# 1a. (Optional) Download per-station obs CSVs from CMADaaS
python download_obs.py

# 1b. (Only for new raw obs data) Convert per-station CSVs to ob{year}.npy
python extract_obs.py

# 2. Convert CSV files to npy arrays
python convert_npy.py

# 3. Train the LightGBM model
python lgb_uv.py
# or
python lgb_wind.py

# 4. Generate predictions
python lgb_predict.py

# 5. Evaluate performance
python access.py
# or
python access97.py
# or
python access97_uv.py
# or
python access97_sc.py
```

### Configuration

Paths are managed through `config/config.yaml` and local override files:

- `config/config.yaml` - shared constants and path placeholders
- `config/config.local.windows.yaml` - Windows-specific paths
- `config/config.local.linux.yaml` - Linux-specific paths

Copy the local example and fill in the real paths for your machine. These local files are ignored by Git.

If `meb.read_station()` fails with micaps3 format errors (pandas >= 2), see <https://github.com/nmcdev/meteva/issues/193> for the fix.

## Project Structure

```
wind_10m/
├── config/                  # YAML configuration files
├── download_obs.py          # Download per-station obs CSVs from CMADaaS
├── extract_ec.py            # Extract EC forecast data to stations
├── extract_obs.py           # Convert raw per-station obs CSVs to npy
├── convert_npy.py           # Convert CSV files to npy arrays
├── lgb_uv.py                # LightGBM UV-component training
├── lgb_wind.py              # LightGBM wind-magnitude training
├── lgb_cv.py                # LightGBM cross-validation
├── lgb_sc.py                # LightGBM SC-encoding training variant
├── lgb_predict.py           # LightGBM prediction
├── pdfm.py / pdfm_uv.py     # PDF-matching post-processing models
├── avt.py, cnn.py, dnn.py, emd.py, gn.py, ots.py, transformer.py
│                            # PyTorch demo models
├── access.py                # Evaluation (legacy station set)
├── access97.py              # Evaluation (97 stations)
├── access97_uv.py           # Evaluation (97 stations, UV components)
├── access97_sc.py           # Evaluation (97 stations, SC-encoded forecasts)
├── rename_vars.py           # Rename variables in Python source files
├── models/                  # Reusable model modules
│   └── base_torch.py        # Shared torch training utilities
├── evaluation/              # Reusable plotting modules
│   ├── access97_plots.py
│   ├── access97_sc_plots.py
│   └── access97_uv_plots.py
├── src/                     # Shared utility modules
│   ├── metrics.py           # Shared evaluation metrics (Acc class)
│   └── utils.py
├── README.md
├── README_cn.md
└── AGENTS.md
```

## Reproducing the 2411-station experiment

The 2411-station dataset is already stored as per-station numpy arrays, so the
`extract_ec.py` / `convert_npy.py` / `extract_obs.py` stages are not needed.

```bash
# 0. Environment: C:\ProgramData\miniconda3\envs\meteva\python.exe (Python 3.11)
pip install -r requirements.txt

# 1. Build per-station caches (~6 min, ~30 GB under D:\data\2411\cache)
python build_cache_2411.py

# 2. Train five models over six leave-one-year-out folds (~5 h)
python train_2411.py

# 3. Write the verification tables to docs/results
python verify_2411.py

# 4. Mutual-information matrix and figures
python compute_mi_2411.py
python plot_2411.py
```

Local paths live in `config/config.2411.local.yaml` (ignored by Git); the
shared settings are in `config/config.2411.yaml`. Details and checklists:
`docs/repro-2411_cn.md`.

## License

This is an internal research project. No public license is provided.

## Milestones

### Completed

| Task | Description | Completion Date |
|------|-------------|-----------------|
| Project foundation | Established preprocessing/config/src structure and unified path loading | 2026-04-20 |
| Documentation standardization | Rewrote `AGENTS.md` and `README.md` in English and created `README_cn.md` | 2026-07-21 |
| Code standards and demo formalization | Formalized torch demo models and applied coding standards across the project | 2026-07-22 |
| New raw observation preprocessing | Added `extract_obs.py` to convert per-station raw CSVs directly into `ob{year}.npy` arrays | 2026-09-14 |
| Documentation cleanup | Resolved leftover Git conflict markers in `AGENTS.md`, `README.md`, `README_cn.md` and synced the file structure/commands with the actual repository layout | 2026-09-17 |

### In Progress

| Task | Description |
|------|-------------|
| New-station-set experiment | Repeat the full pipeline on a new, larger raw dataset with a brand-new station set |

### Todo

| Task | Description | Priority |
|------|-------------|----------|
| Add unit tests | Add test cases for core functions | P3 |
| Demo model scheme integration | Refactor demo models to support unified/per-lead-time/per-station schemes | P2 |

## Author

- **Author**: yinlb<yinlb3@foxmail.com>, deepseek-flash

**Last Updated**: 2026-09-17

**[中文版本](README_cn.md)**
