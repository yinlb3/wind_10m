# -*- coding: utf-8 -*-

"""
Founded in 2024-07-27
Modified in 2026-07-21
@author: yinlb
"""
import os
import pathlib

import arrow
import numpy as np

from src import utils


THRESHOLDS = (0.3, 1.6, 3.4, 5.5, 8.0)
EPS = 1e-7


def _interp(data: np.ndarray, is_0_84: bool = False) -> np.ndarray:
    """
    Linearly interpolate NWP fields from 3-hourly to hourly intervals.

    Args:
        data: Input array with a 3-hourly lead-time dimension.
        is_0_84: If True, interpolate the full 0-84 h range; otherwise
            interpolate the first 0-24 h range.

    Returns:
        Array with hourly interpolated values.
    """
    # Operate on a copy to avoid mutating the caller's array.
    out = data.copy()
    if is_0_84:
        # 1. Interpolate the 24 segments of 3-hourly data to hourly.
        for i in range(24):
            out[:, 3 * i + 1, :, :] = (
                out[:, 3 * i, :, :] * 2 / 3
                + out[:, 3 * i + 3, :, :] / 3
            )
            out[:, 3 * i + 2, :, :] = (
                out[:, 3 * i, :, :] / 3
                + out[:, 3 * i + 3, :, :] * 2 / 3
            )
        # 2. Interpolate the transition from hour 72 to 78 and 78 to 84.
        out[:, 73, :, :] = (
            out[:, 72, :, :] * 5 / 6
            + out[:, 78, :, :] / 6
        )
        out[:, 74, :, :] = (
            out[:, 72, :, :] * 2 / 3
            + out[:, 78, :, :] / 3
        )
        out[:, 75, :, :] = (
            out[:, 72, :, :] / 2
            + out[:, 78, :, :] / 2
        )
        out[:, 76, :, :] = (
            out[:, 72, :, :] / 3
            + out[:, 78, :, :] * 2 / 3
        )
        out[:, 77, :, :] = (
            out[:, 72, :, :] / 6
            + out[:, 78, :, :] * 5 / 6
        )
        out[:, 79, :, :] = (
            out[:, 78, :, :] * 5 / 6
            + out[:, 84, :, :] / 6
        )
        out[:, 80, :, :] = (
            out[:, 78, :, :] * 2 / 3
            + out[:, 84, :, :] / 3
        )
        out[:, 81, :, :] = (
            out[:, 78, :, :] / 2
            + out[:, 84, :, :] / 2
        )
        out[:, 82, :, :] = (
            out[:, 78, :, :] / 3
            + out[:, 84, :, :] * 2 / 3
        )
        out[:, 83, :, :] = (
            out[:, 78, :, :] / 6
            + out[:, 84, :, :] * 5 / 6
        )
    else:
        # Interpolate the first 0-24 h range.
        for i in range(8):
            out[:, 3 * i + 1, :, :] = (
                out[:, 3 * i, :, :] * 2 / 3
                + out[:, 3 * i + 3, :, :] / 3
            )
            out[:, 3 * i + 2, :, :] = (
                out[:, 3 * i, :, :] / 3
                + out[:, 3 * i + 3, :, :] * 2 / 3
            )
    return out


def _uv_to_ds(uv: np.ndarray) -> np.ndarray:
    """
    Convert UV wind components to wind speed and direction.

    Args:
        uv: Array whose last dimension holds [U, V] components.

    Returns:
        Array whose last dimension holds [direction, speed].
    """
    out = np.zeros_like(uv, dtype=np.float32) + np.nan
    out[:, :, :, 1] = np.sqrt(
        uv[:, :, :, 0] ** 2 + uv[:, :, :, 1] ** 2
    )
    out[:, :, :, 0] = 180 + np.arctan2(
        uv[:, :, :, 0], uv[:, :, :, 1]
    ) * 180 / np.pi
    return out


def _ots(
    train_nwp: np.ndarray,
    train_obs: np.ndarray,
    test_nwp: np.ndarray,
) -> np.ndarray:
    """
    Apply optimal threshold selection (OTS) to a forecast field.

    The algorithm derives a piecewise linear mapping from raw NWP values
    to corrected values by searching thresholds that maximize the Threat
    Score for each reference observation level.

    Args:
        train_nwp: Raw NWP forecast samples used for training.
        train_obs: Observed samples used as the reference for training.
        test_nwp: Raw NWP forecast samples to be corrected.

    Returns:
        Corrected forecast samples.
    """
    # 1. Reference thresholds are processed from high to low.
    obs_thresholds = list(THRESHOLDS).copy()
    obs_thresholds.reverse()

    threshold = 11.0
    fcst_thresholds = list()
    fcst_upper = 100.0
    obs_upper = 100.0
    for ref_threshold in obs_thresholds:
        if np.sum(train_obs >= ref_threshold) == 0:
            fcst_thresholds.append(ref_threshold)
            fcst_upper = ref_threshold
            obs_upper = ref_threshold
            continue

        # 2. Search the threshold that maximizes the Threat Score.
        ts_scores = list()
        tested_thresholds = list()
        while abs(threshold) > EPS:
            count_a = np.sum(
                ((train_nwp >= threshold) & (train_nwp < fcst_upper))
                & ((train_obs >= ref_threshold) & (train_nwp < obs_upper))
            )
            count_b = np.sum(
                ((train_nwp >= threshold) & (train_nwp < fcst_upper))
                & ((train_obs < ref_threshold) | (train_nwp >= obs_upper))
            )
            count_c = np.sum(
                ((train_nwp < threshold) | (train_nwp >= fcst_upper))
                & ((train_obs >= ref_threshold) & (train_nwp < obs_upper))
            )
            ts_scores.append(
                count_a / (count_a + count_b + count_c + EPS)
            )
            tested_thresholds.append(threshold)
            threshold -= 0.1

        best_idx = int(np.argmax(ts_scores))
        threshold = tested_thresholds[best_idx]
        fcst_thresholds.append(threshold)
        fcst_upper = threshold
        obs_upper = ref_threshold

    # 3. Restore increasing order for both threshold lists.
    fcst_thresholds.reverse()
    obs_thresholds.reverse()

    # 4. Map the test forecast to the corrected values.
    corrected = np.zeros_like(test_nwp, dtype=np.float32) + np.nan
    mask = test_nwp < fcst_thresholds[0]
    corrected[mask] = test_nwp[mask] / fcst_thresholds[0] * obs_thresholds[0]

    for i in range(len(fcst_thresholds) - 1):
        lower = fcst_thresholds[i]
        upper = fcst_thresholds[i + 1]
        lower_obs = obs_thresholds[i]
        upper_obs = obs_thresholds[i + 1]
        mask = (test_nwp >= lower) & (test_nwp < upper)
        corrected[mask] = (
            (test_nwp[mask] - lower) / (upper - lower)
            * (upper_obs - lower_obs)
            + lower_obs
        )

    mask = test_nwp >= fcst_thresholds[-1]
    corrected[mask] = (
        test_nwp[mask] / fcst_thresholds[-1] * obs_thresholds[-1]
    )

    return corrected



def main(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    """
    Run OTS correction for each cross-validation year.

    Args:
        input_path: Directory containing nwp{year}.npy and ob{year}.npy.
        output_path: Directory where the corrected arrays are saved.
    """
    cfg = utils.load_config()
    years = cfg['constants']['years']

    # 1. Load all years of NWP and observation data.
    nwp_years = list()
    obs_years = list()
    for year in years:
        nwp_years.append(np.load(input_path / 'nwp{:d}.npy'.format(year)))
        obs_years.append(np.load(input_path / 'ob{:d}.npy'.format(year)))

    # 2. Cross-validate: leave one year out for testing.
    for i in range(len(nwp_years)):
        year = years[i]
        nwp_train_list = nwp_years.copy()
        nwp_train_list.pop(i)
        obs_train_list = obs_years.copy()
        obs_train_list.pop(i)

        train_data = np.vstack(nwp_train_list)
        train_data = _interp(train_data[:, 12:37, :, :])[:, 1:, :, :]
        train_obs = np.vstack(obs_train_list)
        train_obs = train_obs[:, 13:37, :, 3]
        train_obs[train_obs > 100] = np.nan

        test_data = nwp_years[i]
        test_data = _interp(test_data[:, 12:37, :, :])[:, 1:, :, :]
        test_obs = obs_years[i]
        test_obs = test_obs[:, 13:37, :, 3]
        test_obs[test_obs > 100] = np.nan

        nwp_train_speed = _uv_to_ds(train_data[:, :, :, 3:5])[:, :, :, 1]
        nwp_test_speed = _uv_to_ds(test_data[:, :, :, 3:5])[:, :, :, 1]

        # 3. Scheme 1: one global OTS mapping.
        predict_speed = _ots(nwp_train_speed, train_obs, nwp_test_speed)
        np.save(output_path / f'ob_1_{year}.npy', test_obs)
        np.save(output_path / f'ots_1_{year}.npy', predict_speed)
        np.save(output_path / f'nwp_1_{year}.npy', nwp_test_speed)

        # 4. Scheme 2: per-lead-time OTS mapping.
        predict_speed = np.zeros_like(nwp_test_speed) + np.nan
        for j in range(24):
            predict_speed[:, j, :] = _ots(
                nwp_train_speed[:, j, :],
                train_obs[:, j, :],
                nwp_test_speed[:, j, :],
            )
        np.save(output_path / f'ob_2_{year}.npy', test_obs)
        np.save(output_path / f'ots_2_{year}.npy', predict_speed)
        np.save(output_path / f'nwp_2_{year}.npy', nwp_test_speed)

        # 5. Scheme 3: per-station OTS mapping.
        predict_speed = np.zeros_like(nwp_test_speed) + np.nan
        for j in range(97):
            predict_speed[:, :, j] = _ots(
                nwp_train_speed[:, :, j],
                train_obs[:, :, j],
                nwp_test_speed[:, :, j],
            )
        np.save(output_path / f'ob_3_{year}.npy', test_obs)
        np.save(output_path / f'ots_3_{year}.npy', predict_speed)
        np.save(output_path / f'nwp_3_{year}.npy', nwp_test_speed)

        # 6. Scheme 4: nested lead-time OTS mapping.
        predict_speed = np.zeros_like(nwp_test_speed) + np.nan
        for j in range(24):
            for k in range(24):
                predict_speed[:, j, k] = _ots(
                    nwp_train_speed[:, j, k],
                    train_obs[:, j, k],
                    nwp_test_speed[:, j, k],
                )
        np.save(output_path / f'ob_4_{year}.npy', test_obs)
        np.save(output_path / f'ots_4_{year}.npy', predict_speed)
        np.save(output_path / f'nwp_4_{year}.npy', nwp_test_speed)

        print(f'Finished OTS correction for year {year}.')


if __name__ == '__main__':
    print('Program ots.py started.')
    total_start = arrow.now()

    cfg = utils.load_config()
    input_path = pathlib.Path(cfg['paths']['wind_data'])
    output_path = pathlib.Path(cfg['paths']['project_wind']) / '97'
    os.makedirs(output_path, exist_ok=True)

    main(input_path, output_path)

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(
        'Program ots.py finished, total time '
        f'{utils.format_time(total_elapsed)}.'
    )
