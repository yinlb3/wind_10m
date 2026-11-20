#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Founded in 2023-08-01
Modified in 2026-09-17
@author: yinlb
"""

import pathlib
import sys
import typing

import arrow
import lightgbm as lgb
import numpy as np
from sklearn import model_selection as ms

from src import utils


RANDOM_SEED = 20230718

# ==================== Parameters ====================
# Forecast lead-hour window used for training [hours].
HOUR_BEGIN = 12
HOUR_END = 36
# LightGBM parallel thread count.
LGB_N_JOBS = 20
# Hyperparameter search grid for randomized cross-validation.
PARAM_GRID = {
    'max_depth': (3, 4, 5),
    'num_leaves': (7, 15, 31),
    'subsample': (0.8, 0.9, 1.0),
    'colsample_bytree': (0.8, 0.9, 1.0),
    'reg_alpha': (1, 10, 1000),
    'reg_lambda': (1, 10, 1000),
}
# ====================================================


def _interp(forecast_data: np.ndarray) -> np.ndarray:
    """
    Interpolate 3-hourly forecast data to hourly data.

    Args:
        forecast_data: Input forecast data with shape
            (samples, lead_times, stations, elements).

    Returns:
        np.ndarray: Interpolated forecast data with one fewer lead time.
    """
    for i in range(8):
        forecast_data[:, 3 * i + 1, :, :] = (
            forecast_data[:, 3 * i, :, :] * 2 / 3
            + forecast_data[:, 3 * i + 3, :, :] / 3
        )
        forecast_data[:, 3 * i + 2, :, :] = (
            forecast_data[:, 3 * i, :, :] / 3
            + forecast_data[:, 3 * i + 3, :, :] * 2 / 3
        )
    return forecast_data[:, 1:, :, :]


def _clean(obs_data: np.ndarray) -> np.ndarray:
    """
    Clean observation data by removing invalid wind records.

    Args:
        obs_data: Raw observation data with shape
            (samples, lead_times, stations, elements).

    Returns:
        np.ndarray: Cleaned observation data with invalid values set to NaN.
    """
    no_wind = (obs_data[:, 1:, :, 0] == 999017) & (obs_data[:, 1:, :, 1] == 0)
    dir_valid = obs_data[:, 1:, :, 0] <= 360
    spd_valid = obs_data[:, 1:, :, 1] < 999017
    speed_diff = obs_data[:, 1:, :, 1] - obs_data[:, :-1, :, 1]
    spd_valid &= np.abs(speed_diff) < 10
    dir_valid |= no_wind
    valid_mask = dir_valid & spd_valid
    obs_data = obs_data[:, 1:, :, :]
    obs_data[~valid_mask, :] = np.nan
    return obs_data


def _ds_to_uv(wind_ds: np.ndarray) -> np.ndarray:
    """
    Convert wind direction and speed to U/V wind components.

    Args:
        wind_ds: Wind direction and speed data with shape
            (samples, lead_times, stations, 2).

    Returns:
        np.ndarray: U/V wind components with the same shape as input.
    """
    no_wind = wind_ds[:, :, :, 0] == 999017
    wind_uv = np.zeros_like(wind_ds) + np.nan
    wind_uv[~no_wind, 0] = (
        -wind_ds[~no_wind, 1]
        * np.sin(np.deg2rad(wind_ds[~no_wind, 0]))
    )
    wind_uv[~no_wind, 1] = (
        -wind_ds[~no_wind, 1]
        * np.cos(np.deg2rad(wind_ds[~no_wind, 0]))
    )
    wind_uv[no_wind, :] = 0
    return wind_uv


def _uv_to_ds(wind_uv: np.ndarray) -> np.ndarray:
    """
    Convert U/V wind components to wind direction and speed.

    Args:
        wind_uv: U/V wind components with shape
            (samples, lead_times, stations, 2).

    Returns:
        np.ndarray: Wind direction and speed with the same shape as input.
    """
    wind_ds = np.zeros_like(wind_uv) + np.nan
    wind_ds[:, :, :, 1] = np.sqrt(
        wind_uv[:, :, :, 0] ** 2 + wind_uv[:, :, :, 1] ** 2
    )
    wind_ds[:, :, :, 0] = (
        180
        + np.arctan2(wind_uv[:, :, :, 0], wind_uv[:, :, :, 1]) * 180 / np.pi
    )
    return wind_ds


def main(input_path: str, years: typing.List[int]) -> None:
    """
    Run LightGBM hyperparameter search with randomized CV.

    Args:
        input_path: Directory containing the .npy forecast
            and observation files.
        years: typing.List of years to load for training.
    """
    # 1. Load forecast and observation data for each year.
    input_dir = pathlib.Path(input_path)
    nwp_data = list()
    obs_data = list()
    for year in years:
        nwp_data.append(np.load(input_dir / 'nwp{:d}.npy'.format(year)))
        obs_data.append(np.load(input_dir / 'ob{:d}.npy'.format(year)))
        print(f'Loaded data for year {year}')

    # 2. Stack and preprocess the data.
    nwp_data = np.vstack(nwp_data)
    nwp_data = nwp_data[:, HOUR_BEGIN:HOUR_END + 1, :, :]
    nwp_data = _interp(nwp_data)
    obs_data = np.vstack(obs_data)
    obs_data = obs_data[:, HOUR_BEGIN:HOUR_END + 1, :, 2:4]
    obs_data = _clean(obs_data)
    wind_uv = _ds_to_uv(obs_data)

    # 3. Build valid mask and feature/target arrays.
    valid_mask = ~np.isnan(wind_uv[:, :, :, 0])
    for i in range(nwp_data.shape[-1]):
        valid_mask &= ~np.isnan(nwp_data[:, :, :, i])
    features = nwp_data[valid_mask, :]
    target = wind_uv[:, :, :, 0][valid_mask]
    feat_mean = np.mean(features, axis=0)
    feat_std = np.std(features, axis=0)
    features = (features - feat_mean) / feat_std

    # 4. Train LightGBM with randomized cross-validation search.
    regressor = lgb.LGBMRegressor(
        n_jobs=LGB_N_JOBS,
        verbose=0,
        force_col_wise=True,
        random_state=RANDOM_SEED,
    )
    rand_search = ms.RandomizedSearchCV(regressor, PARAM_GRID, verbose=2)
    rand_search.fit(features, target)
    print(f'Best CV score: {rand_search.best_score_}')
    print(f'Best parameters: {rand_search.best_params_}')


if __name__ == '__main__':
    print('Program lgb_cv.py started')
    total_start = arrow.now()

    cfg = utils.load_config()
    if len(sys.argv) == 1:
        main(cfg['paths']['wind_data'], cfg['constants']['years'])
    elif len(sys.argv) == 2:
        main(sys.argv[1], cfg['constants']['years'])

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(
        'Program lgb_cv.py finished, total time '
        '{:s}'.format(utils.format_time(total_elapsed))
    )

