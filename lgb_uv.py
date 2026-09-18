#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Founded in 2023-07-18
Modified in 2026-09-17
@author: yinlb, deepseek-flash
"""

import os
import pathlib
import sys
import typing

import arrow
import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn import linear_model as lm

from src import utils


RANDOM_SEED = 20230718

# ==================== Parameters ====================
# Forecast lead-hour window used for training and evaluation [hours].
HOUR_BEGIN = 12
HOUR_END = 36
# LightGBM parallel thread count.
LGB_N_JOBS = -1
# ====================================================


def _interp(data: np.ndarray, is_0_84: bool = False) -> np.ndarray:
    """
    Interpolate 3-hourly forecast fields to hourly values.

    For the normal 24-hour window (12-36 h) the input is assumed to have
    25 hourly slots (indices 0-24) and indices 1-24 are returned. For the
    extended 0-84 h window, the input is assumed to have 85 slots and
    indices 1-84 are returned.

    Args:
        data: Input array with shape (samples, hours, stations, features).
        is_0_84: Whether to process the 0-84 h extended window.

    Returns:
        Hourly interpolated array.
    """
    if is_0_84:
        for i in range(24):
            prev = data[:, 3 * i, :, :]
            next_ = data[:, 3 * i + 3, :, :]
            data[:, 3 * i + 1, :, :] = prev * 2 / 3 + next_ / 3
            data[:, 3 * i + 2, :, :] = prev / 3 + next_ * 2 / 3
        # Interpolate between hours 72 and 78.
        for target, weight in (
            (73, 5 / 6), (74, 2 / 3), (75, 1 / 2),
            (76, 1 / 3), (77, 1 / 6)
        ):
            data[:, target, :, :] = (
                data[:, 72, :, :] * weight
                + data[:, 78, :, :] * (1 - weight)
            )
        # Interpolate between hours 78 and 84.
        for target, weight in (
            (79, 5 / 6), (80, 2 / 3), (81, 1 / 2),
            (82, 1 / 3), (83, 1 / 6)
        ):
            data[:, target, :, :] = (
                data[:, 78, :, :] * weight
                + data[:, 84, :, :] * (1 - weight)
            )
        return data[:, 1:, :, :]

    for i in range(8):
        prev = data[:, 3 * i, :, :]
        next_ = data[:, 3 * i + 3, :, :]
        data[:, 3 * i + 1, :, :] = prev * 2 / 3 + next_ / 3
        data[:, 3 * i + 2, :, :] = prev / 3 + next_ * 2 / 3
    return data[:, 1:, :, :]


def _clean(data: np.ndarray) -> np.ndarray:
    """
    Clean observation wind data and remove invalid samples.

    Samples with direction 999017 and speed below 0.3 m/s are treated as
    calm wind. Sudden speed changes larger than 10 m/s between consecutive
    times are flagged as invalid.

    Args:
        data: Raw observation array with shape (..., hours, stations, 4).

    Returns:
        Cleaned observation array with the same trailing dimensions.
    """
    no_wind = (data[:, 1:, :, 0] == 999017) & (data[:, 1:, :, 1] < 0.3)
    valid_dir = data[:, 1:, :, 0] <= 360
    valid_speed = data[:, 1:, :, 1] < 999017
    delta = data[:, 1:, :, 1] - data[:, :-1, :, 1]
    valid_speed &= np.abs(delta) < 10
    valid_dir |= no_wind
    valid = valid_dir & valid_speed
    data = data[:, 1:, :, :]
    data[~valid, :] = np.nan
    data[no_wind, 0] = 999017
    return data


def _ds_to_uv(ds: np.ndarray) -> np.ndarray:
    """
    Convert wind direction/speed to U/V components.

    Calm wind samples (direction == 999017) are preserved with their original
    speed and a special marker for the U component.

    Args:
        ds: Wind direction/speed array with trailing dimension size 2.

    Returns:
        Array containing U/V wind components.
    """
    no_wind = ds[:, :, :, 0] == 999017
    uv = np.zeros_like(ds, dtype=np.float32) + np.nan
    uv[~no_wind, 0] = -ds[~no_wind, 1] * np.sin(np.deg2rad(ds[~no_wind, 0]))
    uv[~no_wind, 1] = -ds[~no_wind, 1] * np.cos(np.deg2rad(ds[~no_wind, 0]))
    uv[no_wind, 0] = 999017
    uv[no_wind, 1] = ds[no_wind, 1]
    return uv


def _uv_to_ds(uv: np.ndarray) -> np.ndarray:
    """
    Convert U/V wind components back to direction/speed.

    Calm wind samples are reconstructed from the U component marker. Wind
    speeds below 0.3 m/s are clamped to 0.3 m/s for non-calm samples.

    Args:
        uv: Array containing U/V components with trailing dimension size 2.

    Returns:
        Wind direction/speed array.
    """
    ds = np.zeros_like(uv, dtype=np.float32) + np.nan
    # Condition 1: calm-wind marker
    no_wind = uv[:, :, :, 0] == 999017
    wind_speed = np.zeros_like(uv[:, :, :, 1])
    wind_speed[no_wind] = uv[no_wind, 1]
    wind_speed[~no_wind] = np.sqrt(
        uv[~no_wind, 0] ** 2 + uv[~no_wind, 1] ** 2
    )
    # Condition 2: speed is indeed below 0.3
    no_wind |= wind_speed < 0.3
    # Calm flag but speed >= 0.3: push back to grade 0
    wind_speed[no_wind & (wind_speed >= 0.3)] = 0.29
    # Defensive: not calm but < 0.3, raise to grade 1
    wind_speed[~no_wind & (wind_speed < 0.3)] = 0.3
    ds[:, :, :, 1] = wind_speed
    ds[no_wind, 0] = 999017
    ds[~no_wind, 0] = 180 + np.arctan2(
        uv[~no_wind, 0], uv[~no_wind, 1]
    ) * 180 / np.pi
    return ds


def _train(
    data: np.ndarray, label: np.ndarray, model_name: str
) -> typing.List:
    """
    Train U and V regression models for a single data split.

    Args:
        data: Training features with shape (samples, hours, stations, features).
        label: Training targets with shape (samples, hours, stations, 2).
        model_name: Model type, either 'lgb' or 'mlr'.

    Returns:
        typing.List of two fitted models (U component, V component).

    Raises:
        AssertionError: If model_name is neither 'lgb' nor 'mlr'.
    """
    assert model_name in ('lgb', 'mlr')
    valid = ~np.isnan(label[:, :, :, 0])
    valid &= ~np.isnan(label[:, :, :, 1])
    for i in range(data.shape[-1]):
        valid &= ~np.isnan(data[:, :, :, i])
    features = data[valid, :]
    targets = label[valid, :]
    models = list()
    is_calm = targets[:, 0] == 999017
    u_target = targets[~is_calm, 0]
    if model_name == 'lgb':
        model = lgbm.LGBMRegressor(
            n_jobs=LGB_N_JOBS, verbose=-1, force_col_wise=True,
            random_state=RANDOM_SEED
        )
    elif model_name == 'mlr':
        model = lm.LinearRegression()
    model.fit(features[~is_calm], u_target)
    models.append(model)
    v_target = targets[~is_calm, 1]
    if model_name == 'lgb':
        model = lgbm.LGBMRegressor(
            n_jobs=LGB_N_JOBS, verbose=-1, force_col_wise=True,
            random_state=RANDOM_SEED
        )
    elif model_name == 'mlr':
        model = lm.LinearRegression()
    model.fit(features[~is_calm], v_target)
    models.append(model)
    return models


def _train2(
    data: np.ndarray,
    label: np.ndarray,
    option: int,
    model_name: str
) -> typing.List:
    """
    Train models according to the chosen strategy option.

    Args:
        data: Training features with shape (samples, hours, stations, features).
        label: Training targets with shape (samples, hours, stations, 2).
        option: Strategy selector (1: unified, 2: per hour, 3: per station,
            4: per hour and station).
        model_name: Model type, either 'lgb' or 'mlr'.

    Returns:
        Nested list of trained models.
    """
    models = list()
    n_samples, n_hours, n_stations, n_features = data.shape
    if option == 1:
        models = _train(data, label, model_name)
    elif option == 2:
        for i in range(n_hours):
            temp_data = np.reshape(
                data[:, i, :, :],
                shape=(n_samples, 1, n_stations, n_features)
            )
            temp_label = np.reshape(
                label[:, i, :, :],
                shape=(n_samples, 1, n_stations, 2)
            )
            models.append(_train(temp_data, temp_label, model_name))
    elif option == 3:
        for i in range(n_stations):
            temp_data = np.reshape(
                data[:, :, i, :],
                shape=(n_samples, n_hours, 1, n_features)
            )
            temp_label = np.reshape(
                label[:, :, i, :],
                shape=(n_samples, n_hours, 1, 2)
            )
            models.append(_train(temp_data, temp_label, model_name))
    elif option == 4:
        for i in range(n_hours):
            models.append(list())
            for j in range(n_stations):
                temp_data = np.reshape(
                    data[:, i, j, :],
                    shape=(n_samples, 1, 1, n_features)
                )
                temp_label = np.reshape(
                    label[:, i, j, :],
                    shape=(n_samples, 1, 1, 2)
                )
                models[i].append(_train(temp_data, temp_label, model_name))
    return models


def _test(models: typing.List, data: np.ndarray) -> np.ndarray:
    """
    Predict U/V components using the trained models.

    Args:
        models: typing.List of two fitted models (U, V).
        data: Input features with the same shape used during training.

    Returns:
        Predicted U/V components with NaN for invalid input samples.
    """
    valid = np.ones(data.shape[:-1], dtype=bool)
    for i in range(data.shape[-1]):
        valid &= ~np.isnan(data[..., i])
    prediction = np.zeros(
        list(data.shape[:-1]) + [2], dtype=np.float32
    ) + np.nan
    features = data[valid, :]
    comp_u = models[0].predict(features)
    comp_v = models[1].predict(features)
    uv = np.stack([comp_u, comp_v], axis=1)
    prediction[valid] = uv
    return prediction


def _test2(
    models: typing.List, data: np.ndarray, option: int
) -> np.ndarray:
    """
    Predict U/V components following the chosen model strategy.

    Args:
        models: Nested list of trained models from _train2.
        data: Input features.
        option: Strategy selector (1-4).

    Returns:
        Predicted U/V components.
    """
    n_samples, n_hours, n_stations, n_features = data.shape
    prediction = np.zeros(
        (n_samples, n_hours, n_stations, 2), dtype=np.float32
    ) + np.nan
    if option == 1:
        prediction = _test(models, data)
    elif option == 2:
        for i in range(n_hours):
            prediction[:, i, :, :] = _test(models[i], data[:, i, :, :])
    elif option == 3:
        for i in range(n_stations):
            prediction[:, :, i, :] = _test(models[i], data[:, :, i, :])
    elif option == 4:
        for i in range(n_hours):
            for j in range(n_stations):
                prediction[:, i, j, :] = _test(
                    models[i][j], data[:, i, j, :]
                )
    return prediction


def main(
    input_path: str,
    model_path: str,
    output_path: str,
    station_info_path: str
) -> None:
    """
    Train UV correction models and generate predictions for each year.

    Args:
        input_path: Directory containing NWP and observation .npy files.
        model_path: Reserved model directory (currently unused).
        output_path: Directory where output .npy files will be saved.
        station_info_path: pathlib.Path to the station information CSV file.
    """
    # 1. Load configuration and prepare the input/output directories.
    cfg = utils.load_config()
    years = cfg['constants']['years']
    os.makedirs(output_path, exist_ok=True)
    input_dir = pathlib.Path(input_path)
    output_dir = pathlib.Path(output_path)

    # 2. Load and sort the station metadata.
    station_info = pd.read_csv(
        station_info_path, encoding='gb2312', low_memory=False
    )
    station_info.sort_values(by=['台站号'], inplace=True)
    station_info.reset_index(drop=True, inplace=True)

    # 3. Load NWP and observation data for every year.
    data_list = list()
    label_list = list()
    for year in years:
        data_list.append(np.load(input_dir / f'nwp{year}.npy'))
        label_list.append(np.load(input_dir / f'ob{year}.npy'))

    # 4. Leave-one-year-out cross-validation over all years.
    lgb_minutes = {1: list(), 2: list(), 3: list(), 4: list()}
    mlr_minutes = {1: list(), 2: list(), 3: list(), 4: list()}
    for i in range(len(years)):
        # 5. Build the training and test split for the held-out year.
        year = years[i]
        data_list2 = data_list.copy()
        data_list2.pop(i)
        label_list2 = label_list.copy()
        label_list2.pop(i)
        train_data = np.vstack(data_list2)[:, HOUR_BEGIN:HOUR_END + 1, :, :]
        train_data = _interp(train_data)
        train_label = np.vstack(label_list2)
        train_label = train_label[:, HOUR_BEGIN:HOUR_END + 1, :, 2:4]
        train_label = _clean(train_label)
        train_uv = _ds_to_uv(train_label)
        test_data = data_list[i][:, HOUR_BEGIN:HOUR_END + 1, :, :]
        test_data = _interp(test_data)
        test_label = label_list[i]
        test_label = test_label[:, HOUR_BEGIN:HOUR_END + 1, :, 2:4]
        test_label = _clean(test_label)
        np.save(output_dir / f'ob_{year}.npy', test_label)
        nwp_uv = test_data[:, :, :, 3:5]
        nwp_label = _uv_to_ds(nwp_uv)
        np.save(output_dir / f'nwp_{year}.npy', nwp_label)

        # 6. Train and predict for every aggregation option.
        for j in range(4):
            # 7. Skip every option except the per-station one (option 3).
            if j != 2:
                continue

            # 8. Train LightGBM and save its corrected predictions.
            time_arrow = arrow.now()
            models = _train2(train_data, train_uv, j + 1, 'lgb')
            lgb_minutes[j + 1].append(
                (arrow.now() - time_arrow).total_seconds() / 60
            )
            predict_uv = _test2(models, test_data, j + 1)
            predict_label = _uv_to_ds(predict_uv)
            np.save(output_dir / f'lgb_{j + 1}_{year}_uv.npy', predict_uv)
            np.save(output_dir / f'lgb_{j + 1}_{year}.npy', predict_label)

            # 9. Standardize the features per aggregation option.
            train_x = np.copy(train_data)
            test_x = np.copy(test_data)
            n_samples, n_hours, n_stations, n_features = train_x.shape
            if j == 0:
                data_mean = np.nanmean(train_data, axis=(0, 1, 2))
                data_std = np.nanstd(train_data, axis=(0, 1, 2))
                train_x = (train_x - data_mean) / data_std
                test_x = (test_x - data_mean) / data_std
            elif j == 1:
                for k in range(n_hours):
                    data_mean = np.nanmean(
                        train_data[:, k, :, :], axis=(0, 1)
                    )
                    data_std = np.nanstd(
                        train_data[:, k, :, :], axis=(0, 1)
                    )
                    train_x[:, k, :, :] = (
                        train_x[:, k, :, :] - data_mean
                   ) / data_std
                    test_x[:, k, :, :] = (
                        test_x[:, k, :, :] - data_mean
                   ) / data_std
            elif j == 2:
                for k in range(n_stations):
                    data_mean = np.nanmean(
                        train_data[:, :, k, :], axis=(0, 1)
                    )
                    data_std = np.nanstd(
                        train_data[:, :, k, :], axis=(0, 1)
                    )
                    train_x[:, :, k, :] = (
                        train_x[:, :, k, :] - data_mean
                   ) / data_std
                    test_x[:, :, k, :] = (
                        test_x[:, :, k, :] - data_mean
                   ) / data_std
            elif j == 3:
                for k in range(n_hours):
                    for sta_idx in range(n_stations):
                        data_mean = np.nanmean(
                            train_data[:, k, sta_idx, :], axis=0
                        )
                        data_std = np.nanstd(
                            train_data[:, k, sta_idx, :], axis=0
                        )
                        train_x[:, k, sta_idx, :] = (
                            train_x[:, k, sta_idx, :] - data_mean
                       ) / data_std
                        test_x[:, k, sta_idx, :] = (
                            test_x[:, k, sta_idx, :] - data_mean
                       ) / data_std

            # 10. Train the MLR baseline and save its predictions.
            time_arrow = arrow.now()
            models = _train2(train_x, train_uv, j + 1, 'mlr')
            mlr_minutes[j + 1].append(
                (arrow.now() - time_arrow).total_seconds() / 60
            )
            predict_uv = _test2(models, test_x, j + 1)
            np.save(output_dir / f'mlr_{j + 1}_{year}_uv.npy', predict_uv)
            predict_label = _uv_to_ds(predict_uv)
            np.save(output_dir / f'mlr_{j + 1}_{year}.npy', predict_label)
        print(f'Training and prediction for {year} completed.')

    # pd.DataFrame(lgb).to_csv(output_dir / 'lgb_time.csv', index=False)
    # pd.DataFrame(mlr).to_csv(output_dir / 'mlr_time.csv', index=False)


if __name__ == '__main__':
    print('Program lgb_uv.py started')
    total_start = arrow.now()

    cfg = utils.load_config()
    if len(sys.argv) == 1:
        main(
            cfg['paths']['wind_data'],
            cfg['paths']['wind2_model'],
            str(pathlib.Path(cfg['paths']['project_wind']) / '97-uv'),
            cfg['paths']['station_csv']
        )
    elif len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(
        f'Program lgb_uv.py finished, total time '
        f'{utils.format_time(total_elapsed)}'
    )
