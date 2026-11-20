# -*- coding: utf-8 -*-
"""
Founded in 2024-07-27
Modified in 2026-09-17
@author: yinlb
"""

import os
import pathlib
import sys
import typing

import arrow
import joblib
import lightgbm as lgbm
import numpy as np
import pandas as pd

from src import utils


RANDOM_SEED = 20230718
cfg = utils.load_config()


def _interp(data: np.ndarray, is_0_84: bool = False) -> np.ndarray:
    """Interpolate forecast data to fill intermediate lead times.

    Linear interpolation between 3-hourly EC forecast lead times to
    produce hourly values for the selected forecast window.

    Args:
        data: Input array with shape (samples, lead_times, stations, features).
        is_0_84: Whether to interpolate the full 0-84 hour range.

    Returns:
        The interpolated array with the same shape as input.
    """
    if is_0_84:
        for i in range(24):
            data[:, 3 * i + 1, :, :] = (
                data[:, 3 * i, :, :] * 2 / 3 +
                data[:, 3 * i + 3, :, :] / 3
            )
            data[:, 3 * i + 2, :, :] = (
                data[:, 3 * i, :, :] / 3 +
                data[:, 3 * i + 3, :, :] * 2 / 3
            )
        data[:, 73, :, :] = data[:, 72, :, :] * 5 / 6 + data[:, 78, :, :] / 6
        data[:, 74, :, :] = data[:, 72, :, :] * 2 / 3 + data[:, 78, :, :] / 3
        data[:, 75, :, :] = data[:, 72, :, :] / 2 + data[:, 78, :, :] / 2
        data[:, 76, :, :] = data[:, 72, :, :] / 3 + data[:, 78, :, :] * 2 / 3
        data[:, 77, :, :] = data[:, 72, :, :] / 6 + data[:, 78, :, :] * 5 / 6
        data[:, 79, :, :] = data[:, 78, :, :] * 5 / 6 + data[:, 84, :, :] / 6
        data[:, 80, :, :] = data[:, 78, :, :] * 2 / 3 + data[:, 84, :, :] / 3
        data[:, 81, :, :] = data[:, 78, :, :] / 2 + data[:, 84, :, :] / 2
        data[:, 82, :, :] = data[:, 78, :, :] / 3 + data[:, 84, :, :] * 2 / 3
        data[:, 83, :, :] = data[:, 78, :, :] / 6 + data[:, 84, :, :] * 5 / 6
    else:
        for i in range(8):
            data[:, 3 * i + 1, :, :] = (
                data[:, 3 * i, :, :] * 2 / 3 +
                data[:, 3 * i + 3, :, :] / 3
            )
            data[:, 3 * i + 2, :, :] = (
                data[:, 3 * i, :, :] / 3 +
                data[:, 3 * i + 3, :, :] * 2 / 3
            )
    return data


def _uv_to_ds(uv_data: np.ndarray) -> np.ndarray:
    """Convert U/V wind components to direction and speed.

    Args:
        uv_data: Array with U and V wind components in the last dimension.

    Returns:
        Array with wind direction (degrees) and speed (m/s).
    """
    dir_speed = np.zeros_like(uv_data, dtype=np.float32) + np.nan
    dir_speed[:, :, :, 1] = np.sqrt(
        uv_data[:, :, :, 0] ** 2 + uv_data[:, :, :, 1] ** 2
    )
    dir_speed[:, :, :, 0] = 180 + np.arctan2(
        uv_data[:, :, :, 0], uv_data[:, :, :, 1]
    ) * 180 / np.pi
    return dir_speed


def _train(data: np.ndarray, label: np.ndarray) -> lgbm.LGBMRegressor:
    """Train a single LightGBM regressor.

    Args:
        data: Training features, shape (samples, lead_times, stations, features)
        label: Training targets, shape (samples, lead_times, stations).

    Returns:
        Trained LightGBM regressor.
    """
    valid = ~np.isnan(label)
    for i in range(data.shape[-1]):
        valid &= ~np.isnan(data[:, :, :, i])
    features = data[valid, :]
    target = label[valid]
    model = lgbm.LGBMRegressor(
        n_jobs=20, verbose=-1, force_col_wise=True, random_state=RANDOM_SEED
    )
    model.fit(features, target)
    return model


def _train2(data: np.ndarray, label: np.ndarray, option: int) -> typing.List:
    """Train LightGBM regressors for the selected option.

    Options:
        1: single model for all lead times and stations.
        2: one model per lead time.
        3: one model per station.
        4: one model per lead time and station.

    Args:
        data: Training features.
        label: Training targets.
        option: Training strategy (1-4).

    Returns:
        typing.List of trained models.
    """
    models = list()
    n_samples, n_lead, n_stations, n_features = data.shape
    if option == 1:
        models.append(_train(data, label))
    elif option == 2:
        for i in range(n_lead):
            lead_data = np.reshape(
                data[:, i, :, :], shape=(
                    n_samples, 1, n_stations, n_features
                )
            )
            lead_label = np.reshape(
                label[:, i, :], shape=(n_samples, 1, n_stations)
            )
            models.append(_train(lead_data, lead_label))
    elif option == 3:
        for i in range(n_stations):
            sta_data = np.reshape(
                data[:, :, i, :], shape=(n_samples, n_lead, 1, n_features)
            )
            sta_label = np.reshape(
                label[:, :, i], shape=(n_samples, n_lead, 1)
            )
            models.append(_train(sta_data, sta_label))
    elif option == 4:
        for i in range(n_lead):
            models.append(list())
            for j in range(n_stations):
                cell_data = np.reshape(
                    data[:, i, j, :], shape=(n_samples, 1, 1, n_features)
                )
                cell_label = np.reshape(
                    label[:, i, j], shape=(n_samples, 1, 1)
                )
                models[i].append(_train(cell_data, cell_label))
    return models


def _test(model: lgbm.LGBMRegressor, data: np.ndarray) -> np.ndarray:
    """Predict with a single LightGBM regressor.

    Args:
        model: Trained LightGBM regressor.
        data: Input features.

    Returns:
        Predicted values with the same spatial shape as input.
    """
    valid = np.ones_like(data[:, :, :, 0], dtype=np.bool_)
    for i in range(data.shape[-1]):
        valid &= ~np.isnan(data[:, :, :, i])
    pred = np.zeros_like(data[:, :, :, 0], dtype=np.float32) + np.nan
    features = data[valid, :]
    pred[valid] = model.predict(features)
    return pred


def _test2(models: typing.List, data: np.ndarray, option: int) -> np.ndarray:
    """Predict with the selected model configuration.

    Args:
        models: typing.List of trained models.
        data: Input features.
        option: Prediction strategy (1-4).

    Returns:
        Predicted values.
    """
    n_samples, n_lead, n_stations, n_features = data.shape
    pred = np.zeros(
        shape=(n_samples, n_lead, n_stations), dtype=data.dtype
    ) + np.nan
    if option == 1:
        pred = _test(models[0], data)
    elif option == 2:
        for i in range(n_lead):
            model = models[i]
            lead_data = np.reshape(
                data[:, i, :, :], shape=(
                    n_samples, 1, n_stations, n_features
                )
            )
            lead_pred = _test(model, lead_data)
            pred[:, i, :] = np.reshape(
                lead_pred, shape=(n_samples, n_stations)
            )
    elif option == 3:
        for i in range(n_stations):
            model = models[i]
            sta_data = np.reshape(
                data[:, :, i, :], shape=(n_samples, n_lead, 1, n_features)
            )
            sta_pred = _test(model, sta_data)
            pred[:, :, i] = np.reshape(
                sta_pred, shape=(n_samples, n_lead)
            )
    elif option == 4:
        for i in range(n_lead):
            for j in range(n_stations):
                model = models[i][j]
                cell_data = np.reshape(
                    data[:, i, j, :], shape=(n_samples, 1, 1, n_features)
                )
                cell_pred = _test(model, cell_data)
                pred[:, i, j] = np.reshape(cell_pred, shape=n_samples)
    return pred


def _test3(models: typing.List, data: np.ndarray, option: int) -> np.ndarray:
    """Predict with station-wise models applied to all stations.

    This is a variant of option 3 where each station model is used to
    predict every station, producing a cross-station prediction matrix.

    Args:
        models: typing.List of trained station models.
        data: Input features.
        option: Must be 3 for this function.

    Returns:
        Cross-station predictions with shape
        (stations, samples, lead_times, stations).
    """
    n_samples, n_lead, n_stations, n_features = data.shape
    pred = np.zeros(
        shape=(n_stations, n_samples, n_lead, n_stations),
        dtype=data.dtype
    ) + np.nan
    if option == 3:
        for i in range(n_stations):
            model = models[i]
            for j in range(n_stations):
                sta_data = np.reshape(
                    data[:, :, j, :], shape=(
                        n_samples, n_lead, 1, n_features
                    )
                )
                sta_pred = _test(model, sta_data)
                pred[i, :, :, j] = np.reshape(
                    sta_pred, shape=(n_samples, n_lead)
                )
    return pred


def main(
    input_path: str,
    model_path: str,
    output_path: str,
    station_csv_path: str,
) -> None:
    """Run cross-validation prediction for 2017-2022.

    Args:
        input_path: Directory containing nwp{year}.npy and ob{year}.npy files.
        model_path: Directory containing trained model files.
        output_path: Directory where prediction files will be saved.
        station_csv_path: pathlib.Path to the station information CSV.
    """
    # 1. Resolve the paths and ensure the output directory exists.
    input_path = pathlib.Path(input_path)
    model_path = pathlib.Path(model_path)
    output_path = pathlib.Path(output_path)
    station_csv_path = pathlib.Path(station_csv_path)
    os.makedirs(output_path, exist_ok=True)

    # 2. Load and sort the station metadata.
    station_info = pd.read_csv(
        station_csv_path, encoding='gb2312', low_memory=False
    )
    station_info.sort_values(by=['台站号'], inplace=True)
    station_info.reset_index(drop=True, inplace=True)

    # 3. Load NWP and observation data for every year.
    data_list = list()
    label_list = list()
    years = cfg['constants']['years']
    for year in years:
        data_list.append(np.load(
            input_path / 'nwp{:d}.npy'.format(year)
        ))
        label_list.append(np.load(
            input_path / 'ob{:d}.npy'.format(year)
        ))
        print('Loading year:', year)

    # 4. Predict for every held-out year under cross-validation.
    for i in range(4, 6):
        # 5. Build the training and test split for the held-out year.
        year = years[i]
        data_list2 = data_list.copy()
        data_list2.pop(i)
        label_list2 = label_list.copy()
        label_list2.pop(i)
        data = np.vstack(data_list2)
        train_data = data
        train_data = _interp(train_data[:, 12:37, :, :])[:, 1:, :, :]
        train_label = np.vstack(label_list2)
        train_label = train_label[:, 13:37, :, 3]
        train_label[train_label > 100] = np.nan
        data = data_list[i]
        test_data = data
        test_data = _interp(test_data[:, 12:37, :, :])[:, 1:, :, :]
        test_label = label_list[i]
        test_label = test_label[:, 13:37, :, 3]
        test_label[test_label > 100] = np.nan

        # 6. Load the trained model, predict, and save the result.
        models = joblib.load(model_path / f'lgb_{year}.joblib')
        predict_label = _test3(models, test_data, 3)
        np.save(
            output_path / f'lgb_3_{year}_cross.npy',
            predict_label
        )


if __name__ == '__main__':
    print('Program lgb_predict.py started')
    total_start = arrow.now()

    if len(sys.argv) == 1:
        main(
            input_path=cfg['paths']['wind_data'],
            model_path=cfg['paths']['wind5_model'],
            output_path=str(
                pathlib.Path(cfg['paths']['project_wind']) / '97'
            ),
            station_csv_path=cfg['paths']['station_csv'],
        )
    elif len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(
        'Program lgb_predict.py finished, total time {}'.format(
            utils.format_time(total_elapsed)
        )
    )
