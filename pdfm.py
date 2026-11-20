#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Founded in 2024-08-18
Modified in 2026-09-17
@author: yinlb
"""

import copy
import os
import pathlib
import typing

import arrow
import numpy as np

from src import utils


class PDF:
    """
    Probability density function matching model.

    Attributes:
        _table (np.ndarray): Mapping table from observations to forecasts.
    """

    def __init__(self) -> None:
        """Initialize the model with an empty mapping table."""
        self._table = None

    def copy(self) -> 'PDF':
        """Return a deep copy of this model.

        Returns:
            PDF: A deep copy of this instance.
        """
        return copy.deepcopy(self)

    def fit(self, observations: np.ndarray, forecasts: np.ndarray) -> None:
        """Fit the mapping table from observations to forecasts.

        Args:
            observations: Array of observed values.
            forecasts: Array of forecast values.
        """
        observations = observations[~np.isnan(observations)]
        forecasts = forecasts[~np.isnan(forecasts)]
        levels = np.unique(observations)
        levels = np.sort(levels)
        self._table = np.zeros((len(levels), 2), dtype=np.float32)
        self._table[:, 0] = levels
        sorted_fcsts = np.sort(forecasts)
        for i, level in enumerate(levels):
            quantile = np.mean(observations <= level)
            idx = round(quantile * (len(sorted_fcsts) - 1))
            self._table[i, 1] = sorted_fcsts[idx]

    def predict(self, forecasts: np.ndarray) -> typing.Optional[np.ndarray]:
        """Apply the fitted mapping to forecast values.

        Args:
            forecasts: Array of forecast values to correct.

        Returns:
            Corrected array with the same shape, or None if not fitted.
        """
        if self._table is None:
            return None
        shape = forecasts.shape
        flat = np.reshape(forecasts, -1)
        corrected = np.zeros_like(flat) + np.nan
        for i in range(flat.size):
            value = flat[i]
            if np.isnan(value):
                continue
            if value < self._table[0, 1]:
                corrected[i] = self._table[0, 0]
                continue
            if value > self._table[-1, 1]:
                corrected[i] = self._table[-1, 0]
                continue
            left = 0
            right = self._table.shape[0] - 1
            while right - left > 1:
                mid = round((left + right) / 2)
                if value <= self._table[mid, 1]:
                    right = mid
                else:
                    left = mid
            if self._table[right, 1] == self._table[left, 1]:
                corrected[i] = (
                    self._table[right, 0] + self._table[left, 0]
               ) / 2
            else:
                slope = (
                    (self._table[right, 0] - self._table[left, 0])
                    / (self._table[right, 1] - self._table[left, 1])
                )
                corrected[i] = (
                    self._table[left, 0]
                    + slope * (value - self._table[left, 1])
                )
        corrected = np.reshape(corrected, shape)
        return corrected


def _interp(data: np.ndarray, is_0_84: bool = False) -> np.ndarray:
    """Interpolate forecast lead times to 1-hour resolution.

    Args:
        data: Input array with original lead-time dimension.
        is_0_84: If True, interpolate the full 0-84 hour range.

    Returns:
        The interpolated array.
    """
    data = np.copy(data)
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
    """Convert UV wind components to wind speed and direction.

    Args:
        uv: Array with UV components in the last dimension.

    Returns:
        Array with direction and speed in the last dimension.
    """
    ds = np.zeros_like(uv, dtype=np.float32) + np.nan
    ds[:, :, :, 1] = np.sqrt(
        uv[:, :, :, 0] ** 2 + uv[:, :, :, 1] ** 2
    )
    ds[:, :, :, 0] = 180 + np.arctan2(
        uv[:, :, :, 0], uv[:, :, :, 1]
    ) * 180 / np.pi
    return ds


def main() -> None:
    """Run PDFM cross-validation and save corrected forecasts."""
    cfg = utils.load_config()
    input_path = pathlib.Path(cfg['paths']['wind_data'])
    output_path = pathlib.Path(cfg['paths']['project_wind']) / '97'
    os.makedirs(output_path, exist_ok=True)

    years = cfg['constants']['years']
    nwp_template = cfg['templates']['nwp_npy']
    obs_template = cfg['templates']['obs_npy']

    data_list = list()
    label_list = list()
    for year in years:
        data_file = input_path / nwp_template.format(year=year)
        label_file = input_path / obs_template.format(year=year)
        data_list.append(np.load(str(data_file)))
        label_list.append(np.load(str(label_file)))

    n_years = len(years)
    for i in range(n_years):
        year = years[i]
        train_data_list = data_list.copy()
        train_data_list.pop(i)
        train_label_list = label_list.copy()
        train_label_list.pop(i)

        all_data = np.vstack(train_data_list)
        train_data = _interp(all_data[:, 12:37, :, :])[:, 1:, :, :]
        train_label = np.vstack(train_label_list)
        train_label = train_label[:, 13:37, :, 3]
        train_label[train_label > 100] = np.nan

        all_data = data_list[i]
        test_data = _interp(all_data[:, 12:37, :, :])[:, 1:, :, :]
        test_label = label_list[i]
        test_label = test_label[:, 13:37, :, 3]
        test_label[test_label > 100] = np.nan

        nwp_train = _uv_to_ds(train_data[:, :, :, 3:5])[:, :, :, 1]
        nwp_test = _uv_to_ds(test_data[:, :, :, 3:5])[:, :, :, 1]

        # Method 1: global PDFM.
        model = PDF()
        model.fit(train_label, nwp_train)
        predict_label = model.predict(nwp_test)
        np.save(output_path / f'pdfm_1_{year}.npy', predict_label)

        # Method 2: lead-time-specific PDFM.
        predict_label = np.zeros_like(nwp_test) + np.nan
        for j in range(24):
            model = PDF()
            model.fit(train_label[:, j, :], nwp_train[:, j, :])
            predict_label[:, j, :] = model.predict(nwp_test[:, j, :])
        np.save(output_path / f'pdfm_2_{year}.npy', predict_label)

        # Method 3: station-specific PDFM.
        predict_label = np.zeros_like(nwp_test) + np.nan
        for j in range(nwp_test.shape[2]):
            model = PDF()
            model.fit(train_label[:, :, j], nwp_train[:, :, j])
            predict_label[:, :, j] = model.predict(nwp_test[:, :, j])
        np.save(output_path / f'pdfm_3_{year}.npy', predict_label)

        # Method 4: lead-time and station-specific PDFM.
        predict_label = np.zeros_like(nwp_test) + np.nan
        for j in range(24):
            for k in range(nwp_test.shape[2]):
                model = PDF()
                model.fit(train_label[:, j, k], nwp_train[:, j, k])
                predict_label[:, j, k] = model.predict(nwp_test[:, j, k])
        np.save(output_path / f'pdfm_4_{year}.npy', predict_label)

        print(f'Year {year} finished.')


if __name__ == '__main__':
    print('Program pdfm.py started')
    start = arrow.now()

    main()

    elapsed = (arrow.now() - start).total_seconds()
    print(
        'Program pdfm.py finished, total time '
        f'{utils.format_time(elapsed)}'
    )

