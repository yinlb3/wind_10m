# Wind-10m

> 10m wind speed machine learning forecast correction system for ECMWF NWP products.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3%2B-green)]()

A post-processing and bias-correction system for ECMWF 10m wind speed forecasts using LightGBM.

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Milestones](#milestones)
- [License](#license)
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
python preprocessing/extract_ec.py

# 2. Convert CSV files to npy arrays
python preprocessing/convert_npy.py

# 3. Train the LightGBM model
python models/LightGBM_uv.py
# or
python models/LightGBM_wind.py

# 4. Generate predictions
python models/lgb_predict.py

# 5. Evaluate performance
python evaluation/access.py
# or
python evaluation/access97.py
# or
python evaluation/access97_uv.py
```

### Configuration

Paths are managed through `config/config.yaml` and local override files:

- `config/config.yaml` - shared constants and path placeholders
- `config/config.local.windows.yaml` - Windows-specific paths
- `config/config.local.linux.yaml` - Linux-specific paths

Copy the local example and fill in the real paths for your machine. These local files are ignored by Git.

## Project Structure

```
wind_10m/
├── config/                  # YAML configuration files
├── preprocessing/           # EC4AI extraction and CSV → npy conversion
│   ├── extract_ec.py
│   └── convert_npy.py
├── models/                  # Model training, prediction, and post-processing
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
├── evaluation/              # Model evaluation and visualization
│   ├── access.py
│   ├── access97.py
│   └── access97_uv.py
├── src/                     # Shared utility modules
│   └── utils.py
├── README.md
├── README_cn.md
└── AGENTS.md
```

## Milestones

### Completed

| Task | Description | Completion Date |
|------|-------------|-----------------|
| Project foundation | Established preprocessing/config/src structure and unified path loading | 2026-04-20 |
| Documentation standardization | Rewrote `AGENTS.md` and `README.md` in English and created `README_cn.md` | 2026-07-21 |
| Code standards and demo formalization | Formalized torch demo models and applied coding standards across the project | 2026-07-22 |

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

## License

This is an internal research project. No public license is provided.

## Author

- **Author**: yinlb
