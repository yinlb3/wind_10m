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

# 1a. (Optional) Download per-station obs CSVs from CMADaaS
python preprocessing/download_obs.py

# 1b. (Only for new raw obs data) Convert per-station CSVs to ob{year}.npy
python preprocessing/extract_obs.py

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

If `meb.read_station()` fails with micaps3 format errors (pandas >= 2), see <https://github.com/nmcdev/meteva/issues/193> for the fix.

## Project Structure

```
wind_10m/
├── config/                  # YAML configuration files
├── preprocessing/           # CMADaaS download, EC4AI extraction, CSV → npy
│   ├── download_obs.py
│   ├── extract_ec.py
│   ├── extract_obs.py
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

### Todo

| Task | Description | Priority |
|------|-------------|----------|
| Add unit tests | Add test cases for core functions | P3 |
| Demo model scheme integration | Refactor demo models to support unified/per-lead-time/per-station schemes | P2 |

## License

This is an internal research project. No public license is provided.

## Author

- **Author**: yinlb <yinlb3@foxmail.com>

**Last Updated**: 2026-07-31

**[中文版本](README_cn.md)**
