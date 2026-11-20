#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation metrics for 97-station UV-component wind forecasts.

This module implements the Acc class for wind-direction and wind-speed
verification scores, and a main routine that loads model outputs, computes
scores, and draws comparison charts (bars, violin plots, radar plots).

Founded in 2024-10-11
Modified in 2026-09-18
@author: yinlb
"""

import os
import pathlib
import typing

import arrow
import numpy as np

from evaluation import access97_uv_plots
from src import metrics
from src import utils


def _load_data() -> typing.Tuple[
    typing.List[int], pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Load observations and predictions for the four main models.

    Returns:
        typing.Tuple[typing.List[int], pathlib.Path, pathlib.Path,
        pathlib.Path, pathlib.Path, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray]: Evaluation years, the 97-station path, the
        97-uv path, the figure directory, the station csv path, and the
        stacked observation and four-model forecast arrays.
    """
    # 1. Load configuration and data.
    cfg = utils.load_config()
    years = cfg['constants']['years']
    base_path = pathlib.Path(cfg['paths']['project_wind'])
    path_97 = base_path / '97'
    path_97uv = base_path / '97-uv'
    fig_dir = path_97uv / '图'
    os.makedirs(fig_dir, exist_ok=True)
    station_csv = pathlib.Path(cfg['paths']['station_csv'])

    # 1.1 Load 97-station wind-speed data and inject into UV arrays.
    ob = list()
    ec = list()
    pdfm = list()
    mlr = list()
    lgb = list()
    for year in years:
        ob.append(np.load(path_97 / f'ob_1_{year}.npy'))
        ec.append(np.load(path_97 / f'nwp_1_{year}.npy'))
        pdfm.append(np.load(path_97 / f'pdfm_4_{year}.npy'))
        mlr.append(np.load(path_97 / f'mlr_4_{year}.npy'))
        lgb.append(np.load(path_97 / f'lgb_3_{year}.npy'))
    ob = np.vstack(ob)
    ec = np.vstack(ec)
    pdfm = np.vstack(pdfm)
    mlr = np.vstack(mlr)
    lgb = np.vstack(lgb)

    ob2 = list()
    ec2 = list()
    pdfm2 = list()
    mlr2 = list()
    lgb2 = list()
    for year in years:
        ob2.append(np.load(path_97uv / f'ob_{year}.npy'))
        ec2.append(np.load(path_97uv / f'nwp_{year}.npy'))
        pdfm2.append(np.load(path_97uv / f'pdfm_3_{year}.npy'))
        mlr2.append(np.load(path_97uv / f'mlr_3_{year}.npy'))
        lgb2.append(np.load(path_97uv / f'lgb_3_{year}.npy'))
    ob2 = np.vstack(ob2)
    ec2 = np.vstack(ec2)
    pdfm2 = np.vstack(pdfm2)
    mlr2 = np.vstack(mlr2)
    lgb2 = np.vstack(lgb2)
    # ob2[:, :, :, 1] = ob
    # ec2[:, :, :, 1] = ec
    # pdfm2[:, :, :, 1] = pdfm
    # mlr2[:, :, :, 1] = mlr
    # lgb2[:, :, :, 1] = lgb

    return (
        years, path_97, path_97uv, fig_dir, station_csv,
        ob2, ec2, pdfm2, mlr2, lgb2
    )


def _compute_overall_scores(
    ob2: np.ndarray, ec2: np.ndarray, pdfm2: np.ndarray,
    mlr2: np.ndarray, lgb2: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute overall and wind-direction-dependent verification scores.

    Args:
        ob2 (np.ndarray): Observation array of wind speed and direction.
        ec2 (np.ndarray): ECMWF-IFS forecast array.
        pdfm2 (np.ndarray): PDFM forecast array.
        mlr2 (np.ndarray): MOS forecast array.
        lgb2 (np.ndarray): LightGBM forecast array.

    Returns:
        typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Threat-score grades, wind-direction kappa grades, and the wind-speed
        and wind-direction RMSE grouped by wind direction.
    """
    # 2. Compute overall scores and direction-dependent scores.
    ts_grade = np.zeros((2, 4, 8), dtype=np.float32) + np.nan
    wd_kappa_grade = np.zeros((4, 8), dtype=np.float32) + np.nan
    ws_rmse_wd = np.zeros((4, 9), dtype=np.float32) + np.nan
    wd_rmse_wd = np.zeros((4, 9), dtype=np.float32) + np.nan

    models = [
        ('ECMWF-IFS', ec2),
        ('PDFM', pdfm2),
        ('MOS', mlr2),
        ('LightGBM', lgb2),
    ]
    for idx, (name, pred) in enumerate(models):
        print(f'----- {name} -----')
        acc = metrics.Acc(ob2, pred)
        print(f'  Wind speed ME    : {acc.get_ws_me():.4f} m/s')
        print(f'  Wind speed CC     : {acc.get_ws_r():.4f}')
        print(f'  Wind speed MAE   : {acc.get_ws_mae():.4f} m/s')
        print(f'  Wind speed RMSE  : {acc.get_ws_rmse():.4f} m/s')
        print(f'  Wind speed MRE   : {acc.get_ws_mre():.4f}')
        print(f'  Wind speed OA    : {acc.get_ws_oa():.4f}')
        print(f'  Wind speed kappa : {acc.get_ws_kappa():.4f}')
        ts_grade[0, idx, :] = acc.get_ws_ts()[1:]
        ts_grade[1, idx, :] = acc.get_ws_ts2()
        wd_kappa_grade[idx, :] = acc.get_wd_kappa(True)
        print(f'  Wind direction ME   : {acc.get_wd_me():.2f} °')
        print(f'  Wind direction MAE   : {acc.get_wd_mae():.2f} °')
        print(f'  Wind direction RMSE  : {acc.get_wd_rmse():.2f} °')
        print(f'  Wind direction OA    : {acc.get_wd_oa():.4f}')
        print(f'  Wind direction kappa : {acc.get_wd_kappa():.4f}')
        for i, wd_grade in enumerate([2, 1, 0, 7, 6, 5, 4, 3, 2]):
            wd_rmse_wd[idx, i] = acc.get_wd_rmse(wd_grade=wd_grade)
            ws_rmse_wd[idx, i] = acc.get_ws_rmse(wd_grade=wd_grade)
    return ts_grade, wd_kappa_grade, ws_rmse_wd, wd_rmse_wd


def _build_time_indices(
    years: typing.List[int]
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build year, month, and forecast-hour indices for grouping.

    Args:
        years (typing.List[int]): Years covered by the evaluation.

    Returns:
        typing.Tuple[np.ndarray, np.ndarray, np.ndarray]: Year index,
        month index, and forecast-hour index arrays.
    """
    # 3. Build time indices for year/month/forecast-hour grouping.
    n_cases = round(
        (
            arrow.get(str(years[-1] + 1)) - arrow.get(str(years[0]))
        ).total_seconds()
        / 3600 / 12
    )
    year_ind = np.zeros(n_cases, np.int_) - 1
    month_ind = np.zeros(n_cases, np.int_) - 1
    forecast_hour_ind = np.zeros((n_cases, 24), np.int_) - 1
    for i in range(n_cases):
        time_arrow = arrow.get(str(years[0])).shift(hours=12 * i)
        year_ind[i] = time_arrow.datetime.year
        month_ind[i] = time_arrow.shift(hours=44).datetime.month
        for j in range(24):
            forecast_hour_ind[i, j] = (
                time_arrow.shift(hours=13 + j).datetime.hour
            )
    return year_ind, month_ind, forecast_hour_ind


def _compute_yearly_rmse(
    ob2: np.ndarray, ec2: np.ndarray, pdfm2: np.ndarray,
    mlr2: np.ndarray, lgb2: np.ndarray, year_ind: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Compute yearly wind-speed and wind-direction RMSE.

    Args:
        ob2 (np.ndarray): Observation array of wind speed and direction.
        ec2 (np.ndarray): ECMWF-IFS forecast array.
        pdfm2 (np.ndarray): PDFM forecast array.
        mlr2 (np.ndarray): MOS forecast array.
        lgb2 (np.ndarray): LightGBM forecast array.
        year_ind (np.ndarray): Year index of each case.

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: Wind-speed and wind-direction
        RMSE for the four models in each year.
    """
    # 4.1 RMSE by year.
    ws_rmse_year = np.zeros((4, 6), dtype=np.float32) + np.nan
    wd_rmse_year = np.zeros((4, 6), dtype=np.float32) + np.nan
    for i in range(6):
        mask = year_ind == i + 2017
        acc = metrics.Acc(ob2[mask, :, :], ec2[mask, :, :])
        ws_rmse_year[0, i] = acc.get_ws_rmse()
        wd_rmse_year[0, i] = acc.get_wd_rmse()
        acc = metrics.Acc(ob2[mask, :, :], pdfm2[mask, :, :])
        ws_rmse_year[1, i] = acc.get_ws_rmse()
        wd_rmse_year[1, i] = acc.get_wd_rmse()
        acc = metrics.Acc(ob2[mask, :, :], mlr2[mask, :, :])
        ws_rmse_year[2, i] = acc.get_ws_rmse()
        wd_rmse_year[2, i] = acc.get_wd_rmse()
        acc = metrics.Acc(ob2[mask, :, :], lgb2[mask, :, :])
        ws_rmse_year[3, i] = acc.get_ws_rmse()
        wd_rmse_year[3, i] = acc.get_wd_rmse()
    return ws_rmse_year, wd_rmse_year


def _compute_monthly_rmse(
    ob2: np.ndarray, ec2: np.ndarray, pdfm2: np.ndarray,
    mlr2: np.ndarray, lgb2: np.ndarray, month_ind: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Compute monthly wind-speed and wind-direction RMSE.

    Args:
        ob2 (np.ndarray): Observation array of wind speed and direction.
        ec2 (np.ndarray): ECMWF-IFS forecast array.
        pdfm2 (np.ndarray): PDFM forecast array.
        mlr2 (np.ndarray): MOS forecast array.
        lgb2 (np.ndarray): LightGBM forecast array.
        month_ind (np.ndarray): Month index of each case.

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: Wind-speed and wind-direction
        RMSE for the four models in each month.
    """
    # 4.2 RMSE by month.
    ws_rmse_month = np.zeros((4, 12), dtype=np.float32) + np.nan
    wd_rmse_month = np.zeros((4, 12), dtype=np.float32) + np.nan
    for i in range(12):
        mask = month_ind == i + 1
        acc = metrics.Acc(ob2[mask, :, :], ec2[mask, :, :])
        ws_rmse_month[0, i] = acc.get_ws_rmse()
        wd_rmse_month[0, i] = acc.get_wd_rmse()
        acc = metrics.Acc(ob2[mask, :, :], pdfm2[mask, :, :])
        ws_rmse_month[1, i] = acc.get_ws_rmse()
        wd_rmse_month[1, i] = acc.get_wd_rmse()
        acc = metrics.Acc(ob2[mask, :, :], mlr2[mask, :, :])
        ws_rmse_month[2, i] = acc.get_ws_rmse()
        wd_rmse_month[2, i] = acc.get_wd_rmse()
        acc = metrics.Acc(ob2[mask, :, :], lgb2[mask, :, :])
        ws_rmse_month[3, i] = acc.get_ws_rmse()
        wd_rmse_month[3, i] = acc.get_wd_rmse()
    i = int(np.argmin(ws_rmse_month[3, :]))
    print(
        f'LightGBM monthly wind speed RMSE min: month {i + 1}, '
        f'{ws_rmse_month[3, i]:.4f} m/s'
    )
    i = int(np.argmax(ws_rmse_month[3, :]))
    print(
        f'LightGBM monthly wind speed RMSE max: month {i + 1}, '
        f'{ws_rmse_month[3, i]:.4f} m/s'
    )
    i = int(np.argmin(wd_rmse_month[3, :]))
    print(
        f'LightGBM monthly wind direction RMSE min: month {i + 1}, '
        f'{wd_rmse_month[3, i]:.2f} °'
    )
    i = int(np.argmax(wd_rmse_month[3, :]))
    print(
        f'LightGBM monthly wind direction RMSE max: month {i + 1}, '
        f'{wd_rmse_month[3, i]:.2f} °'
    )
    return ws_rmse_month, wd_rmse_month


def _compute_leadtime_rmse(
    ob2: np.ndarray, ec2: np.ndarray, pdfm2: np.ndarray,
    mlr2: np.ndarray, lgb2: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Compute forecast-lead-time wind-speed and wind-direction RMSE.

    Args:
        ob2 (np.ndarray): Observation array of wind speed and direction.
        ec2 (np.ndarray): ECMWF-IFS forecast array.
        pdfm2 (np.ndarray): PDFM forecast array.
        mlr2 (np.ndarray): MOS forecast array.
        lgb2 (np.ndarray): LightGBM forecast array.

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: Wind-speed and wind-direction
        RMSE for the four models at each forecast lead time.
    """
    # 4.3 RMSE by forecast lead time.
    ws_rmse_vt = np.zeros((4, 24), dtype=np.float32) + np.nan
    wd_rmse_vt = np.zeros((4, 24), dtype=np.float32) + np.nan
    for i in range(24):
        acc = metrics.Acc(ob2[:, i, :], ec2[:, i, :])
        ws_rmse_vt[0, i] = acc.get_ws_rmse()
        wd_rmse_vt[0, i] = acc.get_wd_rmse()
        acc = metrics.Acc(ob2[:, i, :], pdfm2[:, i, :])
        ws_rmse_vt[1, i] = acc.get_ws_rmse()
        wd_rmse_vt[1, i] = acc.get_wd_rmse()
        acc = metrics.Acc(ob2[:, i, :], mlr2[:, i, :])
        ws_rmse_vt[2, i] = acc.get_ws_rmse()
        wd_rmse_vt[2, i] = acc.get_wd_rmse()
        acc = metrics.Acc(ob2[:, i, :], lgb2[:, i, :])
        ws_rmse_vt[3, i] = acc.get_ws_rmse()
        wd_rmse_vt[3, i] = acc.get_wd_rmse()
    return ws_rmse_vt, wd_rmse_vt


def _compute_forecast_hour_rmse(
    ob2: np.ndarray, ec2: np.ndarray, pdfm2: np.ndarray,
    mlr2: np.ndarray, lgb2: np.ndarray, forecast_hour_ind: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Compute forecast-hour-of-day wind-speed and wind-direction RMSE.

    Args:
        ob2 (np.ndarray): Observation array of wind speed and direction.
        ec2 (np.ndarray): ECMWF-IFS forecast array.
        pdfm2 (np.ndarray): PDFM forecast array.
        mlr2 (np.ndarray): MOS forecast array.
        lgb2 (np.ndarray): LightGBM forecast array.
        forecast_hour_ind (np.ndarray): Forecast-hour index of each case.

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: Wind-speed and wind-direction
        RMSE for the four models at each forecast hour of day.
    """
    # 4.4 RMSE by forecast hour of day.
    ws_rmse_fhour = np.zeros((4, 24), dtype=np.float32) + np.nan
    wd_rmse_fhour = np.zeros((4, 24), dtype=np.float32) + np.nan
    for i in range(24):
        mask = forecast_hour_ind == i
        acc = metrics.Acc(ob2[mask], ec2[mask])
        ws_rmse_fhour[0, i] = acc.get_ws_rmse()
        wd_rmse_fhour[0, i] = acc.get_wd_rmse()
        acc = metrics.Acc(ob2[mask], pdfm2[mask])
        ws_rmse_fhour[1, i] = acc.get_ws_rmse()
        wd_rmse_fhour[1, i] = acc.get_wd_rmse()
        acc = metrics.Acc(ob2[mask], mlr2[mask])
        ws_rmse_fhour[2, i] = acc.get_ws_rmse()
        wd_rmse_fhour[2, i] = acc.get_wd_rmse()
        acc = metrics.Acc(ob2[mask], lgb2[mask])
        ws_rmse_fhour[3, i] = acc.get_ws_rmse()
        wd_rmse_fhour[3, i] = acc.get_wd_rmse()
    i = int(np.argmin(ws_rmse_fhour[3, :]))
    print(
        f'LightGBM forecast-hour wind speed RMSE min: {i} h, '
        f'{ws_rmse_fhour[3, i]:.4f} m/s'
    )
    i = int(np.argmax(ws_rmse_fhour[3, :]))
    print(
        f'LightGBM forecast-hour wind speed RMSE max: {i} h, '
        f'{ws_rmse_fhour[3, i]:.4f} m/s'
    )
    i = int(np.argmin(wd_rmse_fhour[3, :]))
    print(
        f'LightGBM forecast-hour wind direction RMSE min: {i} h, '
        f'{wd_rmse_fhour[3, i]:.2f} °'
    )
    i = int(np.argmax(wd_rmse_fhour[3, :]))
    print(
        f'LightGBM forecast-hour wind direction RMSE max: {i} h, '
        f'{wd_rmse_fhour[3, i]:.2f} °'
    )
    return ws_rmse_fhour, wd_rmse_fhour


def main() -> None:
    """
    Run the full evaluation and plotting workflow.
    """
    # 1. Load configuration and evaluation data.
    (
        years, path_97, path_97uv, fig_dir, station_csv,
        ob2, ec2, pdfm2, mlr2, lgb2
    ) = _load_data()

    # 2. Plot mutual information and compute overall scores.
    mi_matrix = np.load(path_97uv / 'mi.npy')
    access97_uv_plots.plot_mi_heatmap(mi_matrix, fig_dir)

    ts_grade, wd_kappa_grade, ws_rmse_wd, wd_rmse_wd = _compute_overall_scores(
        ob2, ec2, pdfm2, mlr2, lgb2
    )
    access97_uv_plots.plot_grade_bars(ts_grade, wd_kappa_grade, fig_dir)

    # 3. Build time indices for year/month/forecast-hour grouping.
    year_ind, month_ind, forecast_hour_ind = _build_time_indices(years)

    # 4.1 RMSE by year.
    ws_rmse_year, wd_rmse_year = _compute_yearly_rmse(
        ob2, ec2, pdfm2, mlr2, lgb2, year_ind
    )
    access97_uv_plots.plot_yearly_rmse(ws_rmse_year, wd_rmse_year, fig_dir)

    # 4.2 RMSE by month.
    ws_rmse_month, wd_rmse_month = _compute_monthly_rmse(
        ob2, ec2, pdfm2, mlr2, lgb2, month_ind
    )
    access97_uv_plots.plot_monthly_rmse(ws_rmse_month, wd_rmse_month, fig_dir)

    # 4.3 RMSE by forecast lead time.
    ws_rmse_vt, wd_rmse_vt = _compute_leadtime_rmse(
        ob2, ec2, pdfm2, mlr2, lgb2
    )
    access97_uv_plots.plot_leadtime_rmse(ws_rmse_vt, wd_rmse_vt, fig_dir)

    # 4.4 RMSE by forecast hour of day.
    ws_rmse_fhour, wd_rmse_fhour = _compute_forecast_hour_rmse(
        ob2, ec2, pdfm2, mlr2, lgb2, forecast_hour_ind
    )
    access97_uv_plots.plot_forecast_hour_rmse(
        ws_rmse_fhour, wd_rmse_fhour, fig_dir
    )

    # 5. Analyze the station-wise errors.
    access97_uv_plots.analyze_stations(
        ob2, ec2, pdfm2, mlr2, lgb2, station_csv, fig_dir
    )

    # 6. Draw the wind-direction-dependent radar plot.
    access97_uv_plots.plot_radar(ws_rmse_wd, wd_rmse_wd, fig_dir)

    # 7. Compare the LightGBM variants.
    access97_uv_plots.compare_lgb_variants(ob2, path_97uv, years)


def _interp(data: np.ndarray, is_0_84: bool = False) -> np.ndarray:
    """
    Interpolate forecast lead times to hourly values.

    Args:
        data: Input array with lead times along the second axis.
        is_0_84: Whether to apply the 0-84h interpolation scheme.

    Returns:
        Array with interpolated lead times.
    """
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


if __name__ == '__main__':
    print('Program access97_uv.py started')
    total_start = arrow.now()

    main()

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(
        'Program access97_uv.py finished, total time '
        f'{utils.format_time(total_elapsed)}'
    )
