#!user/bin.python3
# -*- coding: utf-8 -*-

"""
Founded in 2023-09-17
Modified in 2026-09-15
@author: yinlb
"""

import os
import pathlib
import sys
import typing

import arrow
import lightgbm as lgbm
import numpy as np
import pandas as pd

from src import utils


RANDOM_SEED = 20230718

# ==================== Parameters ====================
# Forecast lead-hour window used for training and evaluation [hours].
HOUR_BEGIN = 12
HOUR_END = 36
# LightGBM parallel thread count.
LGB_N_JOBS = 20
# ====================================================


def _interp(data: np.ndarray, is_0_84: bool = False) -> np.ndarray:
    """
    Interpolate NWP data to a finer temporal resolution.

    Args:
        data: Input array with shape (samples, times, stations, features).
        is_0_84: If True, interpolate 85 original times to 84 times.

    Returns:
        Array with interpolated temporal dimension.
    """
    result = data.copy()
    if is_0_84:
        for i in range(24):
            result[:, 3 * i + 1, :, :] = (
                result[:, 3 * i, :, :] * 2 / 3
                + result[:, 3 * i + 3, :, :] / 3
            )
            result[:, 3 * i + 2, :, :] = (
                result[:, 3 * i, :, :] / 3
                + result[:, 3 * i + 3, :, :] * 2 / 3
            )
        result[:, 73, :, :] = (
            result[:, 72, :, :] * 5 / 6
            + result[:, 78, :, :] / 6
        )
        result[:, 74, :, :] = (
            result[:, 72, :, :] * 2 / 3
            + result[:, 78, :, :] / 3
        )
        result[:, 75, :, :] = (
            result[:, 72, :, :] / 2
            + result[:, 78, :, :] / 2
        )
        result[:, 76, :, :] = (
            result[:, 72, :, :] / 3
            + result[:, 78, :, :] * 2 / 3
        )
        result[:, 77, :, :] = (
            result[:, 72, :, :] / 6
            + result[:, 78, :, :] * 5 / 6
        )
        result[:, 79, :, :] = (
            result[:, 78, :, :] * 5 / 6
            + result[:, 84, :, :] / 6
        )
        result[:, 80, :, :] = (
            result[:, 78, :, :] * 2 / 3
            + result[:, 84, :, :] / 3
        )
        result[:, 81, :, :] = (
            result[:, 78, :, :] / 2
            + result[:, 84, :, :] / 2
        )
        result[:, 82, :, :] = (
            result[:, 78, :, :] / 3
            + result[:, 84, :, :] * 2 / 3
        )
        result[:, 83, :, :] = (
            result[:, 78, :, :] / 6
            + result[:, 84, :, :] * 5 / 6
        )
    else:
        for i in range(8):
            result[:, 3 * i + 1, :, :] = (
                result[:, 3 * i, :, :] * 2 / 3
                + result[:, 3 * i + 3, :, :] / 3
            )
            result[:, 3 * i + 2, :, :] = (
                result[:, 3 * i, :, :] / 3
                + result[:, 3 * i + 3, :, :] * 2 / 3
            )
    return result


def _uv_to_ds(uv: np.ndarray) -> np.ndarray:
    """
    Convert wind UV components to wind speed and direction.

    Args:
        uv: Array with UV components at the last dimension.

    Returns:
        Array with wind direction (index 0) and speed (index 1).
    """
    ds = np.zeros_like(uv, dtype=np.float32) + np.nan
    ds[:, :, :, 1] = np.sqrt(
        uv[:, :, :, 0] ** 2 + uv[:, :, :, 1] ** 2
    )
    ds[:, :, :, 0] = (
        180
        + np.arctan2(uv[:, :, :, 0], uv[:, :, :, 1]) * 180 / np.pi
    )
    return ds


def _train(
    data: np.ndarray,
    label: np.ndarray,
) -> lgbm.LGBMRegressor:
    """
    Train a LightGBM regressor on valid samples.

    Args:
        data: Feature array with shape (samples, features).
        label: Target array with shape (samples,).

    Returns:
        Trained LightGBM regressor.
    """
    valid_mask = ~np.isnan(label)
    for i in range(data.shape[-1]):
        valid_mask &= ~np.isnan(data[:, :, :, i])
    features = data[valid_mask, :]
    targets = label[valid_mask]
    model = lgbm.LGBMRegressor(
        n_jobs=LGB_N_JOBS,
        verbose=-1,
        force_col_wise=True,
        random_state=RANDOM_SEED,
    )
    model.fit(features, targets)
    return model


def _train2(
    data: np.ndarray,
    label: np.ndarray,
    option: int,
) -> typing.List:
    """
    Train models according to the specified aggregation option.

    Args:
        data: Feature array with shape (samples, times, stations, features).
        label: Target array with shape (samples, times, stations).
        option: Aggregation option (1-4).

    Returns:
        List of trained models or nested lists for option 4.
    """
    models = list()
    n_samples, n_lead_times, n_stations, n_features = data.shape
    if option == 1:
        models.append(_train(data, label))
    elif option == 2:
        for i in range(n_lead_times):
            temp_data = np.reshape(
                data[:, i, :, :],
                shape=(n_samples, 1, n_stations, n_features),
            )
            temp_label = np.reshape(
                label[:, i, :],
                shape=(n_samples, 1, n_stations),
            )
            models.append(_train(temp_data, temp_label))
    elif option == 3:
        for i in range(n_stations):
            temp_data = np.reshape(
                data[:, :, i, :],
                shape=(n_samples, n_lead_times, 1, n_features),
            )
            temp_label = np.reshape(
                label[:, :, i],
                shape=(n_samples, n_lead_times, 1),
            )
            models.append(_train(temp_data, temp_label))
    elif option == 4:
        for i in range(n_lead_times):
            models.append(list())
            for j in range(n_stations):
                temp_data = np.reshape(
                    data[:, i, j, :],
                    shape=(n_samples, 1, 1, n_features),
                )
                temp_label = np.reshape(
                    label[:, i, j],
                    shape=(n_samples, 1, 1),
                )
                models[i].append(_train(temp_data, temp_label))
    return models


def _test(model: lgbm.LGBMRegressor, data: np.ndarray) -> np.ndarray:
    """
    Predict using a trained model on valid samples.

    Args:
        model: Trained LightGBM regressor.
        data: Feature array with shape (samples, times, stations, features).

    Returns:
        Predicted label array with shape (samples, times, stations).
    """
    valid_mask = np.ones_like(data[:, :, :, 0], dtype=np.bool_)
    for i in range(data.shape[-1]):
        valid_mask &= ~np.isnan(data[:, :, :, i])
    label = np.zeros_like(data[:, :, :, 0], dtype=np.float32) + np.nan
    features = data[valid_mask, :]
    label[valid_mask] = model.predict(features)
    return label


def _test2(
    models: typing.List,
    data: np.ndarray,
    option: int,
) -> np.ndarray:
    """
    Predict using models trained with the specified option.

    Args:
        models: List of trained models or nested lists for option 4.
        data: Feature array with shape (samples, times, stations, features).
        option: Aggregation option (1-4).

    Returns:
        Predicted label array with shape (samples, times, stations).
    """
    n_samples, n_lead_times, n_stations, n_features = data.shape
    label = np.zeros(
        shape=(n_samples, n_lead_times, n_stations),
        dtype=data.dtype,
    ) + np.nan
    if option == 1:
        label = _test(models[0], data)
    elif option == 2:
        for i in range(n_lead_times):
            model = models[i]
            temp_data = np.reshape(
                data[:, i, :, :],
                shape=(n_samples, 1, n_stations, n_features),
            )
            temp_label = _test(model, temp_data)
            label[:, i, :] = np.reshape(
                temp_label,
                shape=(n_samples, n_stations),
            )
    elif option == 3:
        for i in range(n_stations):
            model = models[i]
            temp_data = np.reshape(
                data[:, :, i, :],
                shape=(n_samples, n_lead_times, 1, n_features),
            )
            temp_label = _test(model, temp_data)
            label[:, :, i] = np.reshape(
                temp_label,
                shape=(n_samples, n_lead_times),
            )
    elif option == 4:
        for i in range(n_lead_times):
            for j in range(n_stations):
                model = models[i][j]
                temp_data = np.reshape(
                    data[:, i, j, :],
                    shape=(n_samples, 1, 1, n_features),
                )
                temp_label = _test(model, temp_data)
                label[:, i, j] = np.reshape(
                    temp_label,
                    shape=n_samples,
                )
    return label


def main(
    input_path: str,
    model_path: str,
    output_path: str,
    station_info_path: str,
    years: typing.List[int],
) -> None:
    """
    Train LightGBM models with leave-one-year-out cross-validation.

    Args:
        input_path: Directory containing preprocessed NWP and obs .npy files.
        model_path: Directory for model checkpoints (reserved for future use).
        output_path: Directory where prediction files will be saved.
        station_info_path: Path to the station information CSV.
        years: List of years used for cross-validation.
    """
    # 1. Ensure the output directory exists.
    os.makedirs(output_path, exist_ok=True)

    # 2. Load station metadata.
    station_info = pd.read_csv(
        station_info_path,
        encoding='gb2312',
        low_memory=False,
    )
    station_info.sort_values(by=['台站号'], inplace=True)
    station_info.reset_index(drop=True, inplace=True)

    # 3. Load NWP and observation data for each year.
    data_list = list()
    label_list = list()
    for year in years:
        nwp_file = pathlib.Path(input_path) / 'nwp{:d}.npy'.format(year)
        obs_file = pathlib.Path(input_path) / 'ob{:d}.npy'.format(year)
        data_list.append(np.load(str(nwp_file)))
        label_list.append(np.load(str(obs_file)))

    # 4. Leave-one-year-out cross-validation.
    elapsed_times = np.zeros(4, dtype=np.float32)
    for i in range(len(years)):
        year = years[i]
        data_list_copy = data_list.copy()
        data_list_copy.pop(i)
        label_list_copy = label_list.copy()
        label_list_copy.pop(i)

        train_data = np.vstack(data_list_copy)
        train_data = _interp(
            train_data[:, HOUR_BEGIN:HOUR_END + 1, :, :]
        )[:, 1:, :, :]
        train_label = np.vstack(label_list_copy)
        train_label = train_label[:, HOUR_BEGIN + 1:HOUR_END + 1, :, 3]
        train_label[train_label > 100] = np.nan

        test_data = data_list[i]
        test_data = _interp(
            test_data[:, HOUR_BEGIN:HOUR_END + 1, :, :]
        )[:, 1:, :, :]
        test_label = label_list[i]
        test_label = test_label[:, HOUR_BEGIN + 1:HOUR_END + 1, :, 3]
        test_label[test_label > 100] = np.nan

        nwp_train_label = _uv_to_ds(train_data[:, :, :, 3:5])[:, :, :, 1]
        nwp_test_label = _uv_to_ds(test_data[:, :, :, 3:5])[:, :, :, 1]

        # 5. Train and predict with four aggregation options.
        for j in range(4):
            time_arrow = arrow.now()
            train_x = np.copy(train_data)
            test_x = np.copy(test_data)
            (
                n_train_samples,
                n_lead_times,
                n_stations,
                n_features,
            ) = train_x.shape
            n_test_samples = test_x.shape[0]

            models = _train2(train_x, train_label, j + 1)
            elapsed_times[j] += (
                arrow.now() - time_arrow
            ).total_seconds() / 60
            predict_label = _test2(models, test_data, j + 1)
            output_file = pathlib.Path(output_path) / f'lgb_{j + 1}_{year}.npy'
            np.save(str(output_file), predict_label)
        print(f'Training and prediction for {year} completed.')
    print(f'Training time per option (minutes): {elapsed_times}')


if __name__ == '__main__':
    print('Program lgb_wind.py started')
    total_start = arrow.now()
    cfg = utils.load_config()

    if len(sys.argv) == 1:
        main(
            input_path=str(cfg['paths']['wind_data']),
            model_path=str(cfg['paths']['wind5_model']),
            output_path=str(
                pathlib.Path(cfg['paths']['project_wind']) / '97'
            ),
            station_info_path=str(cfg['paths']['station_csv']),
            years=cfg['constants']['years'],
        )
    elif len(sys.argv) == 5:
        main(
            input_path=sys.argv[1],
            model_path=sys.argv[2],
            output_path=sys.argv[3],
            station_info_path=sys.argv[4],
            years=cfg['constants']['years'],
        )
    else:
        print(
            'Usage: python lgb_wind.py '
            '[input_path model_path output_path station_info_path]'
        )
        sys.exit(1)

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(
        'Program lgb_wind.py finished, total time {}'.format(
            utils.format_time(total_elapsed)
        )
    )
