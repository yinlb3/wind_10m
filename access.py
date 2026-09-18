#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Founded in 2023-07-19
Modified in 2026-09-17
@author: yinlb, deepseek-flash
"""

import datetime
import os
import pathlib
import sys
import typing

import arrow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from meteva import base as meb
from meteva import method as mem
from scipy import stats

from src import utils


METRIC_COLUMNS = (
    'R_EC', 'ME_EC', 'MAE_EC', 'RMSE_EC', 'MRE_EC',
    'R_LightGBM', 'ME_LightGBM', 'MAE_LightGBM',
    'RMSE_LightGBM', 'MRE_LightGBM',
)


def _draw_scatter(
    obs_data: np.ndarray,
    fcst_data: np.ndarray,
    save_path: str,
) -> None:
    """
    Draw a density scatter plot of observations versus forecasts.

    Args:
        obs_data: Observation values.
        fcst_data: Forecast values.
        save_path: File path where the figure will be saved.
    """
    # Calculate the point density.
    stacked_data = np.vstack([obs_data, fcst_data])
    density = stats.gaussian_kde(stacked_data)(stacked_data)

    # Sort points by density so densest points are plotted last.
    sort_index = density.argsort()
    obs_data = obs_data[sort_index]
    fcst_data = fcst_data[sort_index]
    density = density[sort_index]
    print(np.sum(sort_index))

    figure, axes = plt.subplots()
    plt.scatter(obs_data, fcst_data, c=density, s=20, cmap='Spectral')
    plt.colorbar()
    plt.savefig(save_path, dpi=300)
    plt.cla()


def _load_forecasts(
    input_path: str,
    years: typing.List[int],
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load observation, LightGBM, and NWP forecast arrays for all years.

    Args:
        input_path: Directory containing ob_<year>.npy, lgb_<year>.npy,
            and nwp_<year>.npy files.
        years: Years to load.

    Returns:
        Tuple of (observation, LightGBM forecast, NWP forecast) arrays.
    """
    obs_list = list()
    lgb_list = list()
    nwp_list = list()
    for year in years:
        year_data = np.load(str(pathlib.Path(input_path) / f'ob_{year:d}.npy'))
        if len(year_data.shape) == 4:
            year_data = year_data[:, :, :, 1]
        obs_list.append(year_data)

        year_data = np.load(str(pathlib.Path(input_path) / f'lgb_{year:d}.npy'))
        if len(year_data.shape) == 4:
            year_data = year_data[:, :, :, 1]
        lgb_list.append(year_data)

        year_data = np.load(str(pathlib.Path(input_path) / f'nwp_{year:d}.npy'))
        if len(year_data.shape) == 4:
            year_data = year_data[:, :, :, 1]
        nwp_list.append(year_data)
    return np.vstack(obs_list), np.vstack(lgb_list), np.vstack(nwp_list)


def _build_eval_dataframe(
    obs_all: np.ndarray,
    lgb_all: np.ndarray,
    nwp_all: np.ndarray,
    station_data: pd.DataFrame,
    years: typing.List[int],
    forecast: typing.Dict[str, int],
    station_num: int,
) -> pd.DataFrame:
    """
    Build a unified evaluation DataFrame from loaded arrays.

    Args:
        obs_all: Observation array.
        lgb_all: LightGBM forecast array.
        nwp_all: NWP forecast array.
        station_data: Station metadata DataFrame.
        years: Years covered by the loaded arrays.
        forecast: Lead-time configuration, requires the keys
            'ec_start_hour' and 'ec_end_hour'.
        station_num: Number of stations in the arrays.

    Returns:
        Evaluation DataFrame with valid samples only.
    """
    # Two forecast cycles per day over the configured years.
    n_cycles = round(
        (arrow.get(str(years[-1] + 1)) - arrow.get(str(years[0]))).days * 2
    )
    time_array = np.zeros_like(obs_all, dtype='datetime64[ns]')
    for cycle_idx in range(n_cycles):
        time_array[cycle_idx, :, :] = (
            arrow.get(str(years[0])).shift(hours=12 * cycle_idx).datetime
        )
    lead_time_array = np.zeros_like(obs_all, dtype=np.int_)
    n_lead_times = forecast['ec_end_hour'] - forecast['ec_start_hour']
    for lead_idx in range(n_lead_times):
        lead_time_array[:, lead_idx, :] = (
            forecast['ec_start_hour'] + lead_idx
        )
    station_id_array = np.zeros_like(obs_all, dtype=np.int_)
    elev_array = np.zeros_like(obs_all, dtype=np.float_)
    for station_idx in range(station_num):
        station_id_array[:, :, station_idx] = station_data.loc[
            station_idx, 'id'
        ]
        elev_array[:, :, station_idx] = station_data.loc[station_idx, 'e']
    eval_data = meb.sta_data(
        df=pd.DataFrame(
            data={
                'time': np.reshape(time_array, -1),
                'dtime': np.reshape(lead_time_array, -1),
                'id': np.reshape(station_id_array, -1),
                'ob': np.reshape(obs_all, -1),
                'NWP': np.reshape(nwp_all, -1),
                'LightGBM': np.reshape(lgb_all, -1),
            },
        ),
        columns=['time', 'dtime', 'id', 'ob', 'NWP', 'LightGBM'],
    )
    for station_idx in range(station_num):
        eval_data.loc[
            eval_data.loc[:, 'id'] == station_data.loc[station_idx, 'id'],
            'lon',
        ] = station_data.loc[station_idx, 'lon']
        eval_data.loc[
            eval_data.loc[:, 'id'] == station_data.loc[station_idx, 'id'],
            'lat',
        ] = station_data.loc[station_idx, 'lat']
    meb.set_stadata_coords(eval_data, level=0)
    valid_mask = (
        ~np.isnan(eval_data.loc[:, 'ob'])
        & ~np.isnan(eval_data.loc[:, 'NWP'])
        & ~np.isnan(eval_data.loc[:, 'LightGBM'])
    )
    eval_data = eval_data.loc[valid_mask]
    eval_data.reset_index(drop=True, inplace=True)
    print(len(eval_data))
    print(eval_data.max())
    return eval_data


def _append_metric_row(
    metrics_dict: typing.Dict[str, typing.List],
    group_key: str,
    label: str,
    subset_data: pd.DataFrame,
) -> None:
    """
    Compute verification metrics for a subset and append to the dict.

    Args:
        metrics_dict: Dictionary collecting metric columns.
        group_key: Name of the grouping column.
        label: Group label.
        subset_data: Subset evaluation DataFrame.
    """
    obs_array = np.array(subset_data.loc[:, 'ob']).T
    fcst_array = np.array(
        subset_data.loc[:, ('NWP', 'LightGBM')]
    ).T
    corr = mem.corr(obs_array, fcst_array)
    mean_err = mem.me(obs_array, fcst_array)
    abs_err = mem.mae(obs_array, fcst_array)
    rmse_val = mem.rmse(obs_array, fcst_array)
    rel_err = mem.mre(obs_array, fcst_array)
    metrics_dict[group_key].append(label)
    metrics_dict['R_EC'].append(corr[0])
    metrics_dict['ME_EC'].append(mean_err[0])
    metrics_dict['MAE_EC'].append(abs_err[0])
    metrics_dict['RMSE_EC'].append(rmse_val[0])
    metrics_dict['MRE_EC'].append(rel_err[0])
    metrics_dict['R_LightGBM'].append(corr[1])
    metrics_dict['ME_LightGBM'].append(mean_err[1])
    metrics_dict['MAE_LightGBM'].append(abs_err[1])
    metrics_dict['RMSE_LightGBM'].append(rmse_val[1])
    metrics_dict['MRE_LightGBM'].append(rel_err[1])


def _make_metrics_dict(
    group_key: str,
) -> typing.Dict[str, typing.List]:
    """Return an empty metrics dictionary for the given group key."""
    metrics_dict = {
        group_key: list(),
        'R_EC': list(),
        'ME_EC': list(),
        'MAE_EC': list(),
        'RMSE_EC': list(),
        'MRE_EC': list(),
        'R_LightGBM': list(),
        'ME_LightGBM': list(),
        'MAE_LightGBM': list(),
        'RMSE_LightGBM': list(),
        'MRE_LightGBM': list(),
    }
    return metrics_dict


def _plot_metric_bars(
    metrics_df: pd.DataFrame,
    group_key: str,
    output_path: str,
    prefix: str,
    x_start: float,
    n_groups: int,
    x_tick_start: int,
) -> None:
    """
    Draw grouped bar charts for each metric and save to files.

    Args:
        metrics_df: DataFrame with metric columns.
        group_key: Name of the grouping column.
        output_path: Directory to save figures.
        prefix: Filename prefix.
        x_start: Left edge of the first bar group.
        n_groups: Number of groups.
        x_tick_start: Start value for x-axis ticks.
    """
    for metric in ('R', 'ME', 'MAE', 'RMSE', 'MRE'):
        ec_col = f'{metric}_EC'
        lgb_col = f'{metric}_LightGBM'
        plt.bar(
            np.arange(x_start, x_start + n_groups, 1),
            metrics_df.loc[:, ec_col],
            width=0.4,
            color='blue',
            label='EC',
        )
        plt.bar(
            np.arange(x_start + 0.4, x_start + n_groups + 0.4, 1),
            metrics_df.loc[:, lgb_col],
            width=0.4,
            color='red',
            label='LightGBM',
        )
        plt.xticks(
            range(x_tick_start, x_tick_start + n_groups),
            metrics_df.loc[:, group_key],
        )
        plt.title(metric)
        plt.legend()
        plt.savefig(
            str(pathlib.Path(output_path) / f'{prefix}_{metric.lower()}.png'),
            dpi=300,
        )
        plt.cla()


def _evaluate_by_lead_time(
    eval_data: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Compute and plot verification metrics by forecast lead time.

    Args:
        eval_data: Evaluation DataFrame.
        output_path: Directory to save outputs.
    """
    group_key = '时次'
    metrics_dict = _make_metrics_dict(group_key)
    _append_metric_row(metrics_dict, group_key, 'All', eval_data)
    for lead_idx in range(25):
        subset_data = meb.sele_by_para(eval_data, dtime=12 + lead_idx)
        _append_metric_row(
            metrics_dict, group_key, str(12 + lead_idx), subset_data,
        )
    lead_time_df = pd.DataFrame(
        data=metrics_dict,
        columns=[group_key, *METRIC_COLUMNS],
    )
    _plot_metric_bars(
        lead_time_df,
        group_key,
        output_path,
        prefix='分时次',
        x_start=10.8,
        n_groups=26,
        x_tick_start=11,
    )
    lead_time_df.to_csv(
        str(pathlib.Path(output_path) / '分时次.csv'),
        encoding='gb2312',
        index=False,
    )


def _evaluate_by_month(
    eval_data: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Compute and plot verification metrics by month.

    Args:
        eval_data: Evaluation DataFrame.
        output_path: Directory to save outputs.
    """
    group_key = '月份'
    metrics_dict = _make_metrics_dict(group_key)
    _append_metric_row(metrics_dict, group_key, 'All', eval_data)
    for month_idx in range(12):
        subset_data = meb.sele_by_para(eval_data, month=month_idx + 1)
        _append_metric_row(
            metrics_dict, group_key, str(month_idx + 1), subset_data,
        )
    monthly_df = pd.DataFrame(
        data=metrics_dict,
        columns=[group_key, *METRIC_COLUMNS],
    )
    _plot_metric_bars(
        monthly_df,
        group_key,
        output_path,
        prefix='分月份',
        x_start=-0.2,
        n_groups=13,
        x_tick_start=0,
    )
    monthly_df.to_csv(
        str(pathlib.Path(output_path) / '分月份.csv'),
        encoding='gb2312',
        index=False,
    )


def _evaluate_by_grade(
    eval_data: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Compute and plot verification metrics by observed wind speed grade.

    Args:
        eval_data: Evaluation DataFrame.
        output_path: Directory to save outputs.
    """
    group_key = '量级'
    metrics_dict = _make_metrics_dict(group_key)
    grade_masks = [
        (0.3, 1.6, '1级'),
        (1.6, 3.4, '2级'),
        (3.4, 5.5, '3级'),
        (5.5, 8.0, '4级'),
        (8.0, np.inf, '>=5级'),
    ]
    _append_metric_row(metrics_dict, group_key, 'All', eval_data)
    for lower, upper, label in grade_masks:
        if np.isinf(upper):
            valid_mask = eval_data.loc[:, 'ob'] >= lower
        else:
            valid_mask = (
                (eval_data.loc[:, 'ob'] >= lower)
                & (eval_data.loc[:, 'ob'] < upper)
            )
        subset_data = eval_data.loc[valid_mask]
        _append_metric_row(metrics_dict, group_key, label, subset_data)
    grade_df = pd.DataFrame(
        data=metrics_dict,
        columns=[group_key, *METRIC_COLUMNS],
    )
    _plot_metric_bars(
        grade_df,
        group_key,
        output_path,
        prefix='分量级',
        x_start=-0.2,
        n_groups=6,
        x_tick_start=0,
    )
    grade_df.to_csv(
        str(pathlib.Path(output_path) / '分量级.csv'),
        encoding='gb2312',
        index=False,
    )


def _evaluate_by_station(
    eval_data: pd.DataFrame,
    station_data: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Compute station-level metrics and draw spatial distribution maps.

    Args:
        eval_data: Evaluation DataFrame.
        station_data: Station metadata DataFrame.
        output_path: Directory to save outputs.
    """
    station_scores_df = station_data.copy()
    for col in METRIC_COLUMNS:
        station_scores_df.loc[:, col] = 0.0
    for station_idx in range(station_data.shape[0]):
        subset_data = meb.sele_by_para(
            eval_data, id=station_data.loc[station_idx, 'id']
        )
        obs_array = np.array(subset_data.loc[:, 'ob']).T
        fcst_array = np.array(
            subset_data.loc[:, ('NWP', 'LightGBM')]
        ).T
        corr = mem.corr(obs_array, fcst_array)
        mean_err = mem.me(obs_array, fcst_array)
        abs_err = mem.mae(obs_array, fcst_array)
        rmse_val = mem.rmse(obs_array, fcst_array)
        rel_err = mem.mre(obs_array, fcst_array)
        station_scores_df.loc[station_idx, 'R_EC'] = corr[0]
        station_scores_df.loc[station_idx, 'ME_EC'] = mean_err[0]
        station_scores_df.loc[station_idx, 'MAE_EC'] = abs_err[0]
        station_scores_df.loc[station_idx, 'RMSE_EC'] = rmse_val[0]
        station_scores_df.loc[station_idx, 'MRE_EC'] = rel_err[0]
        station_scores_df.loc[station_idx, 'R_LightGBM'] = corr[1]
        station_scores_df.loc[station_idx, 'ME_LightGBM'] = mean_err[1]
        station_scores_df.loc[station_idx, 'MAE_LightGBM'] = abs_err[1]
        station_scores_df.loc[station_idx, 'RMSE_LightGBM'] = rmse_val[1]
        station_scores_df.loc[station_idx, 'MRE_LightGBM'] = rel_err[1]
    meb.set_stadata_coords(
        station_scores_df,
        level=0,
        time=datetime.datetime(2023, 1, 1),
        dtime=0,
    )
    plot_configs = [
        ('R_EC', meb.cmaps.ts, 'R_EC.png'),
        ('ME_EC', meb.cmaps.me, 'ME_EC.png'),
        ('MAE_EC', meb.cmaps.mae, 'MAE_EC.png'),
        ('RMSE_EC', meb.cmaps.mae, 'RMSE_EC.png'),
        ('MRE_EC', meb.cmaps.ts, 'MRE_EC.png'),
        ('R_LightGBM', meb.cmaps.ts, 'R_LightGBM.png'),
        ('ME_LightGBM', meb.cmaps.me, 'ME_LightGBM.png'),
        ('MAE_LightGBM', meb.cmaps.mae, 'MAE_LightGBM.png'),
        ('RMSE_LightGBM', meb.cmaps.mae, 'RMSE_LightGBM.png'),
        ('MRE_LightGBM', meb.cmaps.ts, 'MRE_LightGBM.png'),
    ]
    for value_col, cmap, filename in plot_configs:
        meb.tool.plot_tools.scatter_sta(
            station_scores_df.loc[
                :,
                ('level', 'time', 'dtime', 'id', 'lon', 'lat', value_col),
            ],
            value_column=0,
            map_extend=[108.65, 114.4, 24.5, 30.25],
            cmap=cmap,
            save_path=str(pathlib.Path(output_path) / filename),
            dpi=300,
        )
    station_scores_df.to_csv(
        str(pathlib.Path(output_path) / '分站点.csv'),
        encoding='gb2312',
        index=False,
    )


def main(
    input_path: str,
    output_path: str,
    station_info_path: str,
) -> None:
    """
    Evaluate LightGBM and NWP wind forecasts against observations.

    Loads observation, NWP, and LightGBM forecast data for the configured
    years, computes verification metrics by lead time, month, wind speed
    grade, and station, and saves tables and figures.

    Args:
        input_path: Directory containing ob_<year>.npy, nwp_<year>.npy,
            and lgb_<year>.npy files.
        output_path: Directory where tables and figures will be saved.
        station_info_path: Path to the CSV file with station metadata.
    """
    # Ensure the output directory exists before writing results.
    os.makedirs(output_path, exist_ok=True)

    cfg = utils.load_config()
    years = cfg['constants']['years']

    # 1. Load station information and build station metadata.
    station_info = pd.read_csv(
        str(station_info_path), encoding='gb2312', low_memory=False
    )
    station_data = meb.sta_data(
        station_info.loc[:, ('台站号', '经度', '纬度', '海拔')],
        columns=('id', 'lon', 'lat', 'e'))
    station_data.sort_values(by=['id'], inplace=True)
    station_data.reset_index(drop=True, inplace=True)

    # 2. Load multi-year observation and forecast data.
    obs_all, lgb_all, nwp_all = _load_forecasts(input_path, years)

    # 3. Build the evaluation DataFrame and remove missing values.
    eval_data = _build_eval_dataframe(
        obs_all, lgb_all, nwp_all, station_data,
        years=years,
        forecast=cfg['forecast'],
        station_num=cfg['constants']['station_num'],
    )

    # 4. Evaluate by forecast lead time.
    _evaluate_by_lead_time(eval_data, output_path)

    # 5. Evaluate by month.
    _evaluate_by_month(eval_data, output_path)

    # 6. Evaluate by wind speed grade.
    _evaluate_by_grade(eval_data, output_path)

    # 7. Evaluate by station and plot spatial maps.
    _evaluate_by_station(eval_data, station_data, output_path)


if __name__ == '__main__':
    print('Program access.py started')
    total_start = arrow.now()

    cfg = utils.load_config()
    input_path = pathlib.Path(cfg['paths']['wind5_model'])
    output_path = pathlib.Path(cfg['paths']['wind5_model']) / '图'
    station_info_path = pathlib.Path(cfg['paths']['station_csv'])
    os.makedirs(output_path, exist_ok=True)

    if len(sys.argv) == 1:
        main(
            input_path=str(input_path),
            output_path=str(output_path),
            station_info_path=str(station_info_path),
        )
    elif len(sys.argv) == 4:
        main(
            input_path=sys.argv[1],
            output_path=sys.argv[2],
            station_info_path=sys.argv[3],
        )

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(
        f'Program access.py finished, total time '
        f'{utils.format_time(total_elapsed)}'
    )
