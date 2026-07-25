---
name: wind-10m
description: 10m wind speed machine learning forecast correction system for ECMWF NWP products.
---

# Wind-10m Agent Guide

> **Reference**: See `user-preferences` skill for:
> - Coding standards
> - Git workflow
> - This document only records project-specific information.

---

## 1. Context 📋

### Your role

- Maintain and extend the 10m wind speed forecast correction pipeline.
- Ensure data preprocessing, model training, prediction, and evaluation scripts remain consistent with the skill rules.
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
├── preprocessing/           # EC4AI extraction and CSV → npy conversion
├── models/                  # Model training, prediction, and post-processing
├── evaluation/              # Model evaluation and visualization
├── src/                     # Shared utilities (utils.py)
├── README.md                # English project overview
├── README_cn.md             # Chinese project overview
└── AGENTS.md                # This guide
```

---

## 2. Commands 📋

### Full pipeline

```bash
# 1. Extract EC data to stations
python preprocessing/extract_ec.py

# 2. Convert CSV to npy
python preprocessing/convert_npy.py

# 3. Train LightGBM model
python models/LightGBM_uv.py
# or python models/LightGBM_wind.py

# 4. Generate predictions
python models/lgb_predict.py

# 5. Evaluate model performance
python evaluation/access.py
# or python evaluation/access97.py
# or python evaluation/access97_uv.py
```

### Before running

- Ensure `config/config.local.windows.yaml` or `config/config.local.linux.yaml` contains valid local paths.
- Install dependencies listed in the README.
- Verify the meteva environment is available.

---

## 3. Testing 📖

No automated unit tests are currently in place. Manual verification steps:

1. Run `python -m py_compile <file>.py` for any modified file.
2. Run a single script end-to-end and check that expected output files are generated.
3. Compare evaluation metrics against previous results; relative differences should be ≤ 0.1%.

---

## 4. Code Style 📖

See `user-preferences` skill for the full coding standard. Project-specific conventions:

- All comments and docstrings are in English.
- External data column names and chart labels may remain in Chinese.
- Configuration paths are loaded from `config/` via `src.utils.load_config()`.

---

## 5. Git Workflow 📖

See `user-preferences` skill for Git workflow. Additional notes:

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
- Ignore skill coding standards and Git workflow rules.

---

## 7. Milestones 📋

### Completed

| Task | Description | Completion Date |
|------|-------------|-----------------|
| Project foundation | Established preprocessing/config/src structure, unified path loading, and verified syntax | 2026-04-20 |
| Documentation standardization | Rewrote `AGENTS.md` and `README.md` in English and created `README_cn.md` | 2026-07-21 |
| Code standards and demo formalization | Formalized torch demo models, applied coding standards across the project, and split oversized evaluation files | 2026-07-22 |

### In Progress

| Task | Description | Notes |
|------|-------------|-------|
| Code review and cleanup | Check code quality of each module and ensure data flow is correct | Focus on lead time matching logic |

### Todo

| Task | Description | Priority |
|------|-------------|----------|
| Add unit tests | Add test cases for core functions | P3 |
| Demo model scheme integration | Refactor demo models to support unified/per-lead-time/per-station schemes | P2 |

**Last Updated**: 2026-07-22
