---
name: wind-10m
description: 10m wind speed machine learning forecast correction system for ECMWF NWP products.
---

# Wind-10m Agent Guide

> **Reference**: This document only records project-specific information.

---

## 1. Context 📋

### Your role

- Maintain and extend the 10m wind speed forecast correction pipeline.
- Ensure data preprocessing, model training, prediction, and evaluation scripts remain consistent with the project's coding standards.
- Keep documentation synchronized with code changes.

### Project knowledge

**Background**

This project post-processes ECMWF numerical weather prediction (NWP) products to correct 10m wind speed forecasts at 97 national meteorological observing stations. It uses LightGBM as the main model and supports UV-component and wind-magnitude training modes.

**Tech Stack**

- Python 3.8+
- numpy, pandas, arrow
- LightGBM, scikit-learn
- meteva (meteorological verification)
- matplotlib, seaborn, scipy
- joblib

**File Structure**

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
├── models/                  # Reusable model modules (base_torch.py)
├── evaluation/              # Reusable plotting modules (access97*_plots.py)
├── src/                     # Shared utilities (utils.py, metrics.py)
├── README.md                # English project overview
├── README_cn.md             # Chinese project overview
└── AGENTS.md                # This guide
```

---

## 2. Commands 📋

### Full pipeline

```bash
# 1. Extract EC data to stations
python extract_ec.py

# 1a. (Optional) Download per-station obs CSVs from CMADaaS
python download_obs.py

# 1b. (Only for new raw obs data) Convert per-station CSVs to ob{year}.npy
python extract_obs.py

# 2. Convert CSV to npy
python convert_npy.py


# 3. Train LightGBM model
python lgb_uv.py
# or python lgb_wind.py

# 4. Generate predictions
python lgb_predict.py

# 5. Evaluate model performance
python access.py
# or python access97.py
# or python access97_uv.py
# or python access97_sc.py
```

### 2411-station pipeline (current experiment)

The 2411-station dataset is already stored as per-station numpy arrays, so the
extraction and CSV-to-npy stages are skipped.

```bash
python build_cache_2411.py     # per-station caches, ~6 min, ~30 GB
python train_2411.py           # 6 folds x 5 models, ~5 h
python verify_2411.py          # tables -> docs/results
python compute_mi_2411.py      # mutual-information matrix
python plot_2411.py            # figures -> docs/results/figures
```

Interpreter: `C:\ProgramData\miniconda3\envs\meteva\python.exe` (Python 3.11).
Shared settings: `config/config.2411.yaml`; local paths:
`config/config.2411.local.yaml` (ignored by Git). Checklist:
`docs/repro-2411_cn.md`.

### Before running

- Ensure `config/config.local.windows.yaml` or `config/config.local.linux.yaml` contains valid local paths.
- Install dependencies listed in the README.
- Verify the meteva environment is available.
- If `meb.read_station()` fails with micaps3 format errors (pandas >= 2), see <https://github.com/nmcdev/meteva/issues/193> for the fix.

---

## 3. Testing 📖

No automated unit tests are currently in place. Manual verification steps:

1. Run `python -m py_compile <file>.py` for any modified file.
2. Run a single script end-to-end and check that expected output files are generated.
3. Compare evaluation metrics against previous results; relative differences should be ≤ 0.1%.

---

## 4. Code Style 📖

Follow the general coding standards: English comments, output, and exception messages; `pathlib` for paths; YAML-managed configuration; Google-style docstrings with type annotations; lines within 80 characters. Project-specific conventions:

- All comments and docstrings are in English.
- External data column names and chart labels may remain in Chinese.
- Configuration paths are loaded from `config/` via `src.utils.load_config()`.

---

## 5. Git Workflow 📖

Use English commit messages with a verb prefix (Add/Fix/Update/Remove/Refactor/Docs/Test/Config). Additional notes:

- Do not commit `config/*.local*.yaml` files.
- Do not run `git commit` or `git push` without explicit user approval.
- Keep README and AGENTS.md synchronized with code changes.

---

## 6. Boundaries 📋

### Always do
- Load paths from `config/` instead of hardcoding them in scripts.
- Run `py_compile` after modifying Python files.
- Update `Modified` dates in file headers when editing files.

### Ask first
- Delete or rename existing model output files.
- Add new runtime dependencies.
- Modify evaluation thresholds or numerical constants.

### Never do
- Commit local configuration files with real paths or credentials.
- Run full end-to-end scripts that overwrite large output files without user confirmation.
- Ignore the coding standards and Git workflow rules described above.

---

## 7. Milestones 📋

### Completed

| Task | Description | Completion Date |
|------|-------------|-----------------|
| Project foundation | Established preprocessing/config/src structure, unified path loading, and verified syntax | 2026-04-20 |
| Documentation standardization | Rewrote `AGENTS.md` and `README.md` in English and created `README_cn.md` | 2026-07-21 |
| Code standards and demo formalization | Formalized torch demo models, applied coding standards across the project, and split oversized evaluation files | 2026-07-22 |
| New raw observation preprocessing | Added `extract_obs.py` (per-station raw CSVs at `paths.sta_raw` → `ob{year}[_suffix].npy`, same layout as `convert_npy.py`; cleaning: direction 360→0, direction [0,360), speed [0,1000), else NaN; max-wind pair stays NaN; Inst_Max maps to gust pair). Added `sta_raw` path and `obs_suffix` constant to config (local Windows value: `dth`). Committed as `45efc52` and pushed to Gitee + GitHub | 2026-09-14 |
| Documentation cleanup | Resolved leftover Git conflict markers in `AGENTS.md`, `README.md`, `README_cn.md`, synced the file structure/commands with the actual repository layout, and removed the VS Code webview startup-race trigger (`window.restoreWindows: none`) | 2026-09-17 |

### In Progress

| Task | Description | Notes |
|------|-------------|-------|
| New-station-set experiment | Repeat the full pipeline on a new, larger raw dataset (`D:\data\sta`, 2472 station CSVs, ~185M records) with a **brand-new station set** (not the original 97) | Blocked on user input: ① new station info CSV with `台站号,经度,纬度` (required by `extract_ec.py` interpolation), ② EC4AI raw data coverage for the new years, ③ exact year range. Planned changes once unblocked: make `extract_ec.py` year range configurable (currently hardcoded 2017-2023 at `extract_ec.py:89-90`); support the obs filename suffix in `lgb_uv.py` / `lgb_wind.py` / evaluation loaders (they read `ob{year}.npy` without suffix, e.g. `lgb_uv.py:353-354`); ensure NWP and obs npy station orders match (both ascending station ID) |

### Todo

| Task | Description | Priority |
|------|-------------|----------|
| Add unit tests | Add test cases for core functions | P3 |
| Demo model scheme integration | Refactor demo models to support unified/per-lead-time/per-station schemes | P2 |

**Last Updated**: 2026-09-17
