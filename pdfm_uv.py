#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Founded in 2024-08-18
Modified in 2026-09-17
@author: yinlb, deepseek-flash
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
    Probability density function matching for univariate correction.

    Builds a lookup table that maps each predicted quantile to the
    corresponding observed value, then applies that table to new predictions.

    Attributes:
        _table (np.ndarray): Mapping table from predictions to observations.
    """

    def __init__(self) -> None:
        """Initialize an empty correction table."""
        self._table = None

    def copy(self) -> 'PDF':
        """Return a deep copy of the correction model."""
        return copy.deepcopy(self)

    def fit(self, obs: np.ndarray, prd: np.ndarray) -> None:
        """
        Build a quantile-to-quantile mapping from observations and predictions.

        Args:
            obs: Observed values with possible NaNs.
            prd: Predicted values with possible NaNs.
        """
        obs = obs[~np.isnan(obs)]
        prd = prd[~np.isnan(prd)]
        sorted_obs = np.unique(obs)
        sorted_obs = np.sort(sorted_obs)
        self._table = np.zeros((len(sorted_obs), 2), dtype=np.float32)
        self._table[:, 0] = sorted_obs
        sorted_prd = np.sort(prd)
        for i, obs_val in enumerate(sorted_obs):
            prob = np.mean(obs <= obs_val)
            idx = round(prob * (len(sorted_prd) - 1))
            self._table[i, 1] = sorted_prd[idx]

    def predict(self, prd: np.ndarray) -> typing.Optional[np.ndarray]:
        """
        Apply the fitted mapping to predicted values.

        Args:
            prd: Predicted values to correct.

        Returns:
            Corrected values with the same shape as the input, or None if the
            model has not been fitted.
        """
        if self._table is None:
            return None
        shape = prd.shape
        prd_in = np.reshape(prd, -1)
        prd_out = np.zeros_like(prd_in) + np.nan
        for i in range(prd_in.size):
            value = prd_in[i]
            if np.isnan(value):
                continue
            if value < self._table[0, 1]:
                prd_out[i] = self._table[0, 0]
                continue
            if value > self._table[-1, 1]:
                prd_out[i] = self._table[-1, 0]
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
                prd_out[i] = (self._table[right, 0] + self._table[left, 0]) / 2
            else:
                slope = (
                    (self._table[right, 0] - self._table[left, 0])
                    / (self._table[right, 1] - self._table[left, 1])
                )
                prd_out[i] = self._table[left, 0] + slope * (
                    value - self._table[left, 1]
                )
        prd_out = np.reshape(prd_out, shape)
        return prd_out


def _interp(data: np.ndarray, is_0_84: bool = False) -> np.ndarray:
    """
    Linearly interpolate 3-hourly forecast data to hourly data.

    Args:
        data: Input array with a 3-hourly lead-time dimension.
        is_0_84: Whether the input covers lead times 0-84 and needs special
            tail interpolation. Defaults to False.

    Returns:
        The interpolated array.
    """
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
                data[:, 3 * i, :, :] * 2 / 3
                + data[:, 3 * i + 3, :, :] / 3
            )
            data[:, 3 * i + 2, :, :] = (
                data[:, 3 * i, :, :] / 3
                + data[:, 3 * i + 3, :, :] * 2 / 3
            )
    return data


def _clean(data: np.ndarray) -> np.ndarray:
    """
    Remove invalid observations based on wind direction and speed jumps.

    Args:
        data: Observations with direction, speed, and other features.

    Returns:
        The cleaned observations.
    """
    no_wind = (data[:, 1:, :, 0] == 999017) & (data[:, 1:, :, 1] < 0.3)
    valid_dir = data[:, 1:, :, 0] <= 360
    valid_spd = data[:, 1:, :, 1] < 999017
    delta = data[:, 1:, :, 1] - data[:, :-1, :, 1]
    valid_spd &= np.abs(delta) < 10
    valid_dir |= no_wind
    mask = valid_dir & valid_spd
    data = data[:, 1:, :, :]
    data[~mask, :] = np.nan
    data[no_wind, 0] = 999017
    return data


def _ds_to_uv(wind: np.ndarray) -> np.ndarray:
    """
    Convert wind direction and speed into U/V wind components.

    Args:
        wind: Array with wind direction in the first channel and wind speed in
            the second channel.

    Returns:
        Array with U and V components.
    """
    no_wind = wind[:, :, :, 0] == 999017
    uv = np.zeros_like(wind, dtype=np.float32) + np.nan
    uv[~no_wind, 0] = -wind[~no_wind, 1] * np.sin(
        np.deg2rad(wind[~no_wind, 0])
    )
    uv[~no_wind, 1] = -wind[~no_wind, 1] * np.cos(
        np.deg2rad(wind[~no_wind, 0])
    )
    uv[no_wind, 0] = 999017
    uv[no_wind, 1] = wind[no_wind, 1]
    return uv


def _uv_to_ds(wind_uv: np.ndarray) -> np.ndarray:
    """
    Convert U/V wind components back to wind direction and speed.

    Args:
        wind_uv: Array with U and V components.

    Returns:
        Array with wind direction and speed.
    """
    wind = np.zeros_like(wind_uv, dtype=np.float32) + np.nan
    no_wind = wind_uv[:, :, :, 0] == 999017
    speed = np.sqrt(wind_uv[no_wind, 0] ** 2 + wind_uv[no_wind, 1] ** 2)
    speed[speed >= 0.3] = 0.29
    wind[no_wind, 1] = speed
    wind[no_wind, 0] = 999017
    speed = np.sqrt(wind_uv[~no_wind, 0] ** 2 + wind_uv[~no_wind, 1] ** 2)
    speed[speed < 0.3] = 0.3
    wind[~no_wind, 1] = speed
    wind[~no_wind, 0] = 180 + np.arctan2(
        wind_uv[~no_wind, 0], wind_uv[~no_wind, 1]
    ) * 180 / np.pi
    return wind


def main(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    """
    Run PDFM cross-validation experiments for wind U/V correction.

    Args:
        input_path: Directory containing the NWP and observation .npy files.
        output_path: Directory where the corrected outputs are saved.
    """
    cfg = utils.load_config()
    years = cfg['constants']['years']
    nwp_files = list()
    obs_files = list()
    for year in years:
        nwp_name = cfg['templates']['nwp_npy'].format(year=year)
        obs_name = cfg['templates']['obs_npy'].format(year=year)
        nwp_files.append(np.load(input_path / nwp_name))
        obs_files.append(np.load(input_path / obs_name))

    for i in range(len(years)):
        year = years[i]
        train_nwp = nwp_files.copy()
        train_nwp.pop(i)
        train_obs = obs_files.copy()
        train_obs.pop(i)
        nwp_train = np.vstack(train_nwp)
        nwp_train = _interp(nwp_train[:, 12:37, :, :])[:, 1:, :, :]
        obs_train = np.vstack(train_obs)
        obs_train = obs_train[:, 12:37, :, 2:4]
        obs_train = _clean(obs_train)
        train_uv = _ds_to_uv(obs_train)
        nwp_test = nwp_files[i]
        nwp_test = _interp(nwp_test[:, 12:37, :, :])[:, 1:, :, :]
        obs_test = obs_files[i]
        obs_test = obs_test[:, 13:37, :, 3]
        obs_test[obs_test > 100] = np.nan
        nwp_train_uv = nwp_train[:, :, :, 3:5]
        nwp_test_uv = nwp_test[:, :, :, 3:5]

        # 1. One model trained on all samples.
        predict_uv = np.zeros_like(nwp_test_uv, dtype=np.float32) + np.nan
        mask = train_uv[:, :, :, 0] != 999017
        model_u = PDF()
        model_u.fit(train_uv[mask, 0], nwp_train_uv[mask, 0])
        model_v = PDF()
        model_v.fit(train_uv[mask, 1], nwp_train_uv[mask, 1])
        predict_uv[:, :, :, 0] = model_u.predict(nwp_test_uv[:, :, :, 0])
        predict_uv[:, :, :, 1] = model_v.predict(nwp_test_uv[:, :, :, 1])
        predict_label = _uv_to_ds(predict_uv)
        np.save(output_path / f'pdfm_1_{year}.npy', predict_label)

        # 2. One model per forecast lead time.
        predict_uv = np.zeros_like(nwp_test_uv, dtype=np.float32) + np.nan
        for j in range(24):
            train_u = train_uv[:, j, :, 0]
            train_v = train_uv[:, j, :, 1]
            nwp_u = nwp_train_uv[:, j, :, 0]
            nwp_v = nwp_train_uv[:, j, :, 1]
            test_u = nwp_test_uv[:, j, :, 0]
            test_v = nwp_test_uv[:, j, :, 1]
            mask = train_u != 999017
            model_u = PDF()
            model_u.fit(train_u[mask], nwp_u[mask])
            model_v = PDF()
            model_v.fit(train_v[mask], nwp_v[mask])
            predict_uv[:, j, :, 0] = model_u.predict(test_u)
            predict_uv[:, j, :, 1] = model_v.predict(test_v)
        predict_label = _uv_to_ds(predict_uv)
        np.save(output_path / f'pdfm_2_{year}.npy', predict_label)

        # 3. One model per station.
        predict_uv = np.zeros_like(nwp_test_uv, dtype=np.float32) + np.nan
        for j in range(nwp_test_uv.shape[2]):
            train_u = train_uv[:, :, j, 0]
            train_v = train_uv[:, :, j, 1]
            nwp_u = nwp_train_uv[:, :, j, 0]
            nwp_v = nwp_train_uv[:, :, j, 1]
            test_u = nwp_test_uv[:, :, j, 0]
            test_v = nwp_test_uv[:, :, j, 1]
            mask = train_u != 999017
            model_u = PDF()
            model_u.fit(train_u[mask], nwp_u[mask])
            model_v = PDF()
            model_v.fit(train_v[mask], nwp_v[mask])
            predict_uv[:, :, j, 0] = model_u.predict(test_u)
            predict_uv[:, :, j, 1] = model_v.predict(test_v)
        predict_label = _uv_to_ds(predict_uv)
        np.save(output_path / f'pdfm_3_{year}.npy', predict_label)

        # 4. One model per lead time and station.
        predict_uv = np.zeros_like(nwp_test_uv, dtype=np.float32) + np.nan
        for j in range(24):
            for k in range(nwp_test_uv.shape[2]):
                train_u = train_uv[:, j, k, 0]
                train_v = train_uv[:, j, k, 1]
                nwp_u = nwp_train_uv[:, j, k, 0]
                nwp_v = nwp_train_uv[:, j, k, 1]
                test_u = nwp_test_uv[:, j, k, 0]
                test_v = nwp_test_uv[:, j, k, 1]
                mask = train_u != 999017
                model_u = PDF()
                model_u.fit(train_u[mask], nwp_u[mask])
                model_v = PDF()
                model_v.fit(train_v[mask], nwp_v[mask])
                predict_uv[:, j, k, 0] = model_u.predict(test_u)
                predict_uv[:, j, k, 1] = model_v.predict(test_v)
        predict_label = _uv_to_ds(predict_uv)
        np.save(output_path / f'pdfm_4_{year}.npy', predict_label)
        print(year)


if __name__ == '__main__':
    print('Program pdfm_uv.py started')
    start = arrow.now()

    cfg = utils.load_config()
    input_path = pathlib.Path(cfg['paths']['wind_data'])
    output_path = pathlib.Path(cfg['paths']['project_wind']) / '97-uv'
    os.makedirs(output_path, exist_ok=True)
    main(input_path, output_path)

    elapsed = (arrow.now() - start).total_seconds()
    print(
        f'Program pdfm_uv.py finished, total time '
        f'{utils.format_time(elapsed)}'
    )
