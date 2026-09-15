# -*- coding: utf-8 -*-

"""
Founded in 2024-08-20
Modified in 2026-09-15
@author: yinlb
"""

import copy
import os
import pathlib

import arrow
import numpy as np
from sklearn import linear_model

from src import utils


class AVT:
    """
    Anomaly-variance transformation for forecast bias correction.

    Adjusts the forecast distribution to match the observed distribution by
    removing the forecast mean and standard deviation, then rescaling with the
    observed mean and standard deviation.
    """

    def __init__(self) -> None:
        """Initialize the AVT transformer with empty statistics."""
        self.ob_mean = None
        self.ob_std = None
        self.pr_mean = None
        self.pr_std = None

    def copy(self) -> 'AVT':
        """Return a deep copy of the transformer."""
        return copy.deepcopy(self)

    def fit(self, ob: np.ndarray, pr: np.ndarray) -> None:
        """
        Fit the transformation statistics from observations and forecasts.

        Args:
            ob: Observed values.
            pr: Forecast values.
        """
        self.ob_mean = np.nanmean(ob)
        self.ob_std = np.nanstd(ob)
        self.pr_mean = np.nanmean(pr)
        self.pr_std = np.nanstd(pr)

    def predict(self, pr0: np.ndarray) -> np.ndarray:
        """
        Apply the fitted transformation to a forecast.

        Args:
            pr0: Forecast values to be corrected.

        Returns:
            Bias-corrected forecast values.
        """
        pr = (
            (pr0 - self.pr_mean) / self.pr_std * self.ob_std
            + self.ob_mean
        )
        return pr


def _interp(data: np.ndarray, is_0_84: bool = False) -> np.ndarray:
    """
    Linearly interpolate 3-hourly data to hourly values.

    Args:
        data: Input array with shape (samples, lead_times, stations, elements).
        is_0_84: Whether to interpolate the full 0-84 hour range.

    Returns:
        The interpolated array.
    """
    # Do not mutate the caller's array (A6).
    data = data.copy()
    if is_0_84:
        for i in range(24):
            data[:, 3 * i + 1, :, :] = (
                data[:, 3 * i, :, :] * 2 / 3
                + data[:, 3 * i + 3, :, :] / 3
            )
            data[:, 3 * i + 2, :, :] = (
                data[:, 3 * i, :, :] / 3
                + data[:, 3 * i + 3, :, :] * 2 / 3
            )
        data[:, 73, :, :] = (
            data[:, 72, :, :] * 5 / 6
            + data[:, 78, :, :] / 6
        )
        data[:, 74, :, :] = (
            data[:, 72, :, :] * 2 / 3
            + data[:, 78, :, :] / 3
        )
        data[:, 75, :, :] = (
            data[:, 72, :, :] / 2
            + data[:, 78, :, :] / 2
        )
        data[:, 76, :, :] = (
            data[:, 72, :, :] / 3
            + data[:, 78, :, :] * 2 / 3
        )
        data[:, 77, :, :] = (
            data[:, 72, :, :] / 6
            + data[:, 78, :, :] * 5 / 6
        )
        data[:, 79, :, :] = (
            data[:, 78, :, :] * 5 / 6
            + data[:, 84, :, :] / 6
        )
        data[:, 80, :, :] = (
            data[:, 78, :, :] * 2 / 3
            + data[:, 84, :, :] / 3
        )
        data[:, 81, :, :] = (
            data[:, 78, :, :] / 2
            + data[:, 84, :, :] / 2
        )
        data[:, 82, :, :] = (
            data[:, 78, :, :] / 3
            + data[:, 84, :, :] * 2 / 3
        )
        data[:, 83, :, :] = (
            data[:, 78, :, :] / 6
            + data[:, 84, :, :] * 5 / 6
        )
    else:
        for i in range(8):
            data[:, 3 * i + 1, :, :] = (
                data[:, 3 * i, :, :] * 2 / 3
                + data[:, 3 * i + 3, :, :] / 3
            )
            data[:, 3 * i + 2, :, :] = (
                data[:, 3 * i, :, :] / 3
                + data[:, 3 * i + 3, :, :] * 2 / 3
            )
    return data


def _uv_to_ds(uv: np.ndarray) -> np.ndarray:
    """
    Convert U/V wind components to wind speed and direction.

    Args:
        uv: Array with U/V components in the last dimension.

    Returns:
        Array with wind direction (degree) and speed (m/s) in the last
        dimension.
    """
    ds = np.zeros_like(uv, dtype=np.float32) + np.nan
    ds[:, :, :, 1] = np.sqrt(
        uv[:, :, :, 0] ** 2 + uv[:, :, :, 1] ** 2
    )
    ds[:, :, :, 0] = 180 + np.arctan2(
        uv[:, :, :, 0], uv[:, :, :, 1]
   ) * 180 / np.pi
    return ds


def main(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    """
    Run AVT correction for each cross-validation fold.

    Args:
        input_path: Directory containing nwp{year}.npy and ob{year}.npy.
        output_path: Directory where corrected outputs are saved.
    """
    # 1. Load all yearly data into memory.
    data_list = list()
    label_list = list()
    years = utils.load_config()['constants']['years']
    for year in years:
        data_list.append(
            np.load(input_path / f'nwp{year:d}.npy')
        )
        label_list.append(
            np.load(input_path / f'ob{year:d}.npy')
        )

    # 2. Leave one year out for testing and train on the remaining years.
    for i in range(len(years)):
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
        nwp_train_label = _uv_to_ds(
            train_data[:, :, :, 3:5]
        )[:, :, :, 1]
        nwp_test_label = _uv_to_ds(
            test_data[:, :, :, 3:5]
        )[:, :, :, 1]
        # Scheme 4: per lead-time and site AVT correction.
        predict_label = np.zeros_like(nwp_test_label) + np.nan
        for j in range(24):
            for k in range(24):
                model = AVT()
                model.fit(
                    train_label[:, j, k], nwp_train_label[:, j, k]
                )
                predict_label[:, j, k] = model.predict(
                    nwp_test_label[:, j, k]
                )
        np.save(output_path / f'avt_3_{year}.npy', predict_label)
        print(year)


if __name__ == '__main__':
    print('Program avt.py started')
    total_start = arrow.now()

    cfg = utils.load_config()
    input_path = pathlib.Path(cfg['paths']['wind_data'])
    output_path = pathlib.Path(cfg['paths']['project_wind']) / '97'
    os.makedirs(output_path, exist_ok=True)
    main(input_path, output_path)

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(
        'Program avt.py finished, total time '
        f'{utils.format_time(total_elapsed)}'
    )

