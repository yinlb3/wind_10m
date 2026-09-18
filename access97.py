# -*- coding: utf-8 -*-
"""
Founded in 2024-07-27
Modified in 2026-09-18
@author: yinlb, deepseek-flash

Evaluation script for 97-station 10 m wind speed forecasts.
"""

import copy
import os
import pathlib
import typing

import arrow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from meteva import base as meb

from evaluation import access97_plots
from src import metrics
from src import utils


def _load_forecasts(
    data_dir: pathlib.Path,
    year_values: list,
) -> tuple:
    """
    Load observation and forecast arrays for all years.

    Args:
        data_dir: Directory containing the npy files.
        year_values: Years to load.

    Returns:
        Tuple of (ob, ec, pdfm, mlr, lgb) arrays.
    """
    ob = list()
    ec = list()
    pdfm = list()
    mlr = list()
    lgb = list()
    for year in year_values:
        ob.append(np.load(data_dir / f'ob_1_{year}.npy'))
        ec.append(np.load(data_dir / f'nwp_1_{year}.npy'))
        pdfm.append(np.load(data_dir / f'pdfm_4_{year}.npy'))
        mlr.append(np.load(data_dir / f'mlr_4_{year}.npy'))
        lgb.append(np.load(data_dir / f'lgb_3_{year}.npy'))
    return (
        np.vstack(ob),
        np.vstack(ec),
        np.vstack(pdfm),
        np.vstack(mlr),
        np.vstack(lgb),
    )


def _evaluate_overall(
    ob: np.ndarray,
    predictions: dict,
) -> None:
    """
    Print overall verification metrics for each model.

    Args:
        ob: Observation array.
        predictions: Dictionary mapping model name to prediction array.
    """
    for name, pred in predictions.items():
        print(name)
        acc = metrics.Acc(ob, pred, has_wd=False)
        print(acc.get_ws_r())
        print(acc.get_ws_mae())
        print(acc.get_ws_rmse())
        print(acc.get_ws_mre())
        print(acc.get_ws_oa())
        print(acc.get_ws_kappa())


def _compute_ts_array(
    ob: np.ndarray,
    predictions: dict,
) -> np.ndarray:
    """
    Compute threat score arrays for all models.

    Args:
        ob: Observation array.
        predictions: Ordered dictionary of model predictions.

    Returns:
        TS array of shape (2, n_models, n_grades).
    """
    n_models = len(predictions)
    ts = np.zeros((2, n_models, 8), dtype=np.float32) + np.nan
    for model_idx, pred in enumerate(predictions.values()):
        acc = metrics.Acc(ob, pred, has_wd=False)
        ts[0, model_idx, :] = acc.get_ws_ts()[1:]
        ts[1, model_idx, :] = acc.get_ws_ts2()
    return ts


def _plot_ts_bars(
    ts: np.ndarray,
    figure_dir: pathlib.Path,
) -> None:
    """
    Draw grouped bar charts of TS by wind grade.

    Args:
        ts: TS array of shape (2, n_models, n_grades).
        figure_dir: Directory to save figures.
    """
    model_colors = ['blue', 'green', 'yellow', 'red']
    model_offsets = [-0.3, -0.1, 0.1, 0.3]
    model_names = ['ECMWF-IFS', 'PDFM', 'MLR', 'LightGBM']

    plt.figure(figsize=(15, 5))
    for model_idx in range(ts.shape[1]):
        plt.bar(
            x=np.arange(8) + model_offsets[model_idx],
            height=ts[0, model_idx, :],
            width=0.2,
            color=model_colors[model_idx],
            label=model_names[model_idx],
        )
    plt.xlim((-1, 8))
    grade_labels = [
        '1级', '2级', '3级', '4级', '5级', '6级', '7级', '8级及以上'
    ]
    plt.xticks(range(8), grade_labels)
    plt.ylim((0, 1))
    plt.yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
    plt.xlabel('风速等级')
    plt.ylabel('TS')
    plt.legend()
    plt.savefig(figure_dir / 'ts_grade.jpg', bbox_inches='tight', dpi=800)
    plt.close()


def _plot_cumulative_ts_bars(
    ts: np.ndarray,
    figure_dir: pathlib.Path,
) -> None:
    """
    Draw grouped bar charts of cumulative TS by threshold.

    Args:
        ts: TS array of shape (2, n_models, n_grades).
        figure_dir: Directory to save figures.
    """
    model_colors = ['blue', 'green', 'yellow', 'red']
    model_offsets = [-0.3, -0.1, 0.1, 0.3]
    model_names = ['ECMWF-IFS', 'PDFM', 'MLR', 'LightGBM']

    plt.figure(figsize=(15, 5))
    for model_idx in range(ts.shape[1]):
        plt.bar(
            x=np.arange(8) + model_offsets[model_idx],
            height=ts[1, model_idx, :],
            width=0.2,
            color=model_colors[model_idx],
            label=model_names[model_idx],
        )
    plt.xlim((-1, 8))
    threshold_labels = [
        '≥0.3m/s', '≥1.6m/s', '≥3.4m/s', '≥5.5m/s',
        '≥8.0m/s', '≥10.8m/s', '≥13.9m/s', '≥17.2m/s'
    ]
    plt.xticks(range(8), threshold_labels)
    plt.ylim((0, 1))
    plt.yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
    plt.xlabel('风速等级')
    plt.ylabel('TS')
    plt.legend()
    plt.savefig(figure_dir / 'ts_grade+.jpg', bbox_inches='tight', dpi=800)
    plt.close()


def _build_time_indices(
    year_values: list,
) -> tuple:
    """
    Build year, month, and forecast hour indices for grouping samples.

    Args:
        year_values: typing.List of years in the dataset.

    Returns:
        Tuple of (year_ind, month_ind, forecast_hour_ind) arrays.
    """
    start_year = year_values[0]
    end_year = year_values[-1] + 1
    num_times = round(
        (arrow.get(str(end_year)) - arrow.get(str(start_year))).total_seconds()
        / 3600 / 12
    )
    year_ind = np.zeros(num_times, np.int_) - 1
    month_ind = np.zeros(num_times, np.int_) - 1
    forecast_hour_ind = np.zeros((num_times, 24), np.int_) - 1
    for time_idx in range(num_times):
        time_arrow = arrow.get(str(start_year)).shift(hours=12 * time_idx)
        year_ind[time_idx] = time_arrow.datetime.year
        month_ind[time_idx] = time_arrow.shift(hours=44).datetime.month
        for lead_idx in range(24):
            forecast_hour_ind[time_idx, lead_idx] = (
                time_arrow.shift(hours=13 + lead_idx).datetime.hour
            )
    return year_ind, month_ind, forecast_hour_ind


def _compute_rmse_improvement(
    ob: np.ndarray,
    ec: np.ndarray,
    pdfm: np.ndarray,
    mlr: np.ndarray,
    lgb: np.ndarray,
) -> tuple:
    """
    Compute RMSE for all models and improvement over ECMWF.

    Args:
        ob: Observation array for the selected subset.
        ec: ECMWF forecast array.
        pdfm: PDFM forecast array.
        mlr: MLR forecast array.
        lgb: LightGBM forecast array.

    Returns:
        Tuple of (rmse_values, rmse_improvement) arrays.
    """
    rmse_values = np.zeros((4,), dtype=np.float32) + np.nan
    rmse_improve = np.zeros((3,), dtype=np.float32) + np.nan
    rmse_values[0] = metrics.Acc(ob, ec, has_wd=False).get_ws_rmse()
    rmse_values[1] = metrics.Acc(ob, pdfm, has_wd=False).get_ws_rmse()
    rmse_improve[0] = (
        (rmse_values[0] - rmse_values[1]) / rmse_values[0] * 100
    )
    rmse_values[2] = metrics.Acc(ob, mlr, has_wd=False).get_ws_rmse()
    rmse_improve[1] = (
        (rmse_values[0] - rmse_values[2]) / rmse_values[0] * 100
    )
    rmse_values[3] = metrics.Acc(ob, lgb, has_wd=False).get_ws_rmse()
    rmse_improve[2] = (
        (rmse_values[0] - rmse_values[3]) / rmse_values[0] * 100
    )
    return rmse_values, rmse_improve


def _evaluate_by_year(
    ob: np.ndarray,
    ec: np.ndarray,
    pdfm: np.ndarray,
    mlr: np.ndarray,
    lgb: np.ndarray,
    year_ind: np.ndarray,
    year_values: list,
    figure_dir: pathlib.Path,
) -> None:
    """
    Compute and plot RMSE improvement by year.

    Args:
        ob: Observation array.
        ec: ECMWF forecast array.
        pdfm: PDFM forecast array.
        mlr: MLR forecast array.
        lgb: LightGBM forecast array.
        year_ind: Year index array.
        year_values: Years to evaluate.
        figure_dir: Directory to save figures.
    """
    n_years = len(year_values)
    rmse_improve_year = np.zeros((3, n_years), dtype=np.float32) + np.nan
    for year_idx, year in enumerate(year_values):
        _, rmse_improve_year[:, year_idx] = _compute_rmse_improvement(
            ob[year_ind == year, :, :],
            ec[year_ind == year, :, :],
            pdfm[year_ind == year, :, :],
            mlr[year_ind == year, :, :],
            lgb[year_ind == year, :, :],
        )
    year_labels = [f'{year}年' for year in year_values]
    access97_plots.plot_rmse_improvement_bars(
        rmse_improve_year,
        figure_dir,
        'rmse_improvement_year.jpg',
        year_labels,
        '年份',
    )


def _evaluate_by_month(
    ob: np.ndarray,
    ec: np.ndarray,
    pdfm: np.ndarray,
    mlr: np.ndarray,
    lgb: np.ndarray,
    month_ind: np.ndarray,
    figure_dir: pathlib.Path,
) -> None:
    """
    Compute and plot RMSE improvement by month.

    Args:
        ob: Observation array.
        ec: ECMWF forecast array.
        pdfm: PDFM forecast array.
        mlr: MLR forecast array.
        lgb: LightGBM forecast array.
        month_ind: Month index array.
        figure_dir: Directory to save figures.
    """
    rmse_improve_month = np.zeros((3, 12), dtype=np.float32) + np.nan
    for month_idx in range(12):
        _, rmse_improve_month[:, month_idx] = _compute_rmse_improvement(
            ob[month_ind == month_idx + 1, :, :],
            ec[month_ind == month_idx + 1, :, :],
            pdfm[month_ind == month_idx + 1, :, :],
            mlr[month_ind == month_idx + 1, :, :],
            lgb[month_ind == month_idx + 1, :, :],
        )
    month_labels = [f'{month + 1}月' for month in range(12)]
    access97_plots.plot_rmse_improvement_bars(
        rmse_improve_month,
        figure_dir,
        'rmse_improvement_month.jpg',
        month_labels,
        '月份',
    )


def _evaluate_by_lead_time(
    ob: np.ndarray,
    ec: np.ndarray,
    pdfm: np.ndarray,
    mlr: np.ndarray,
    lgb: np.ndarray,
    figure_dir: pathlib.Path,
) -> None:
    """
    Compute and plot RMSE improvement by lead time.

    Args:
        ob: Observation array.
        ec: ECMWF forecast array.
        pdfm: PDFM forecast array.
        mlr: MLR forecast array.
        lgb: LightGBM forecast array.
        figure_dir: Directory to save figures.
    """
    rmse_improve_vt = np.zeros((3, 24), dtype=np.float32) + np.nan
    for lead_idx in range(24):
        _, rmse_improve_vt[:, lead_idx] = _compute_rmse_improvement(
            ob[:, lead_idx, :],
            ec[:, lead_idx, :],
            pdfm[:, lead_idx, :],
            mlr[:, lead_idx, :],
            lgb[:, lead_idx, :],
        )
    lead_labels = [str(lead + 1) for lead in range(24)]
    access97_plots.plot_rmse_improvement_bars(
        rmse_improve_vt,
        figure_dir,
        'rmse_improvement_vt.jpg',
        lead_labels,
        '预报时效/h',
    )


def _evaluate_by_forecast_hour(
    ob: np.ndarray,
    ec: np.ndarray,
    pdfm: np.ndarray,
    mlr: np.ndarray,
    lgb: np.ndarray,
    forecast_hour_ind: np.ndarray,
    figure_dir: pathlib.Path,
) -> None:
    """
    Compute and plot RMSE improvement by forecast hour.

    Args:
        ob: Observation array.
        ec: ECMWF forecast array.
        pdfm: PDFM forecast array.
        mlr: MLR forecast array.
        lgb: LightGBM forecast array.
        forecast_hour_ind: Forecast hour index array.
        figure_dir: Directory to save figures.
    """
    rmse_improve_fhour = np.zeros((3, 24), dtype=np.float32) + np.nan
    for hour_idx in range(24):
        _, rmse_improve_fhour[:, hour_idx] = _compute_rmse_improvement(
            ob[forecast_hour_ind == hour_idx, :],
            ec[forecast_hour_ind == hour_idx, :],
            pdfm[forecast_hour_ind == hour_idx, :],
            mlr[forecast_hour_ind == hour_idx, :],
            lgb[forecast_hour_ind == hour_idx, :],
        )
    hour_labels = [f'{hour}:00' for hour in range(24)]
    access97_plots.plot_rmse_improvement_bars(
        rmse_improve_fhour,
        figure_dir,
        'rmse_improvement_fhour.jpg',
        hour_labels,
        '预报时间/UTC',
    )


def _evaluate_by_station(
    ob: np.ndarray,
    ec: np.ndarray,
    pdfm: np.ndarray,
    mlr: np.ndarray,
    lgb: np.ndarray,
    station_csv: pathlib.Path,
    figure_dir: pathlib.Path,
    n_stations: int,
) -> typing.Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute station-level RMSE improvement and draw spatial maps.

    Args:
        ob: Observation array.
        ec: ECMWF forecast array.
        pdfm: PDFM forecast array.
        mlr: MLR forecast array.
        lgb: LightGBM forecast array.
        station_csv: pathlib.Path to station metadata CSV.
        figure_dir: Directory to save figures.
        n_stations: Number of stations.

    Returns:
        Array of station-level RMSE improvements of shape (3, n_stations).
    """
    station_info = pd.read_csv(
        station_csv, encoding='gb2312', low_memory=False
    )
    station_info.sort_values(by=['台站号'], inplace=True)
    station_info.reset_index(drop=True, inplace=True)
    sta = station_info.loc[:, ('台站号', '经度', '纬度')]
    sta = meb.sta_data(sta, columns=['id', 'lon', 'lat'])
    sta.loc[:, 'level'] = 0
    sta.loc[:, 'time'] = '2024-01-01 08:00:00'
    sta.loc[:, 'dtime'] = 0
    rmse_improve_sta = np.zeros((3, n_stations), dtype=np.float32) + np.nan
    for station_idx in range(n_stations):
        _, rmse_improve_sta[:, station_idx] = _compute_rmse_improvement(
            ob[:, :, station_idx],
            ec[:, :, station_idx],
            pdfm[:, :, station_idx],
            mlr[:, :, station_idx],
            lgb[:, :, station_idx],
        )
    cmap, clevs = meb.def_cmap_clevs(
        meb.def_cmap_clevs(meb.cmaps.mae, vmin=-20, vmax=80)[0],
        clevs=[-20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80]
    )
    plot_configs = [
        ('pdfm_rmse_improve_sta.jpg', 0),
        ('mlr_rmse_improve_sta.jpg', 1),
        ('lgb_rmse_improve_sta.jpg', 2),
    ]
    for filename, model_idx in plot_configs:
        sta.loc[:, 'data0'] = rmse_improve_sta[model_idx, :]
        meb.tool.plot_tools.scatter_sta(
            sta0=sta,
            map_extend=[108.65, 114.4, 24.5, 30.25],
            clevs=clevs,
            cmap=cmap,
            title=[''],
            save_path=figure_dir / filename,
            dpi=800
        )
    print(np.min(rmse_improve_sta, axis=1))
    print(np.mean(rmse_improve_sta, axis=1))
    print(np.median(rmse_improve_sta, axis=1))
    print(np.max(rmse_improve_sta, axis=1))
    return rmse_improve_sta, station_info


def main() -> None:
    """
    Evaluate wind speed forecasts and generate figures.
    """
    cfg = utils.load_config()
    data_dir = pathlib.Path(cfg['paths']['project_wind']) / '97'
    figure_dir = pathlib.Path(cfg['paths']['figure_dir'])
    station_csv = pathlib.Path(cfg['paths']['station_csv'])
    year_values = cfg['constants']['years']
    os.makedirs(figure_dir, exist_ok=True)

    # 1. Load multi-year forecast and observation data.
    ob, ec, pdfm, mlr, lgb = _load_forecasts(data_dir, year_values)

    # 2. Print overall verification metrics.
    predictions = {
        'ECMWF-IFS': ec,
        'PDFM': pdfm,
        'MLR': mlr,
        'LightGBM': lgb,
    }
    _evaluate_overall(ob, predictions)

    # 3. Compute threat scores by grade.
    ts = _compute_ts_array(ob, predictions)

    # 4. Plot TS for each wind grade.
    _plot_ts_bars(ts, figure_dir)

    # 5. Plot cumulative TS for each threshold.
    _plot_cumulative_ts_bars(ts, figure_dir)

    # 6. Build time indices for grouping samples.
    year_ind, month_ind, forecast_hour_ind = _build_time_indices(year_values)

    # 7. Compute RMSE improvement by year.
    _evaluate_by_year(
        ob, ec, pdfm, mlr, lgb, year_ind, year_values, figure_dir
    )

    # 8. Compute RMSE improvement by month.
    _evaluate_by_month(
        ob, ec, pdfm, mlr, lgb, month_ind, figure_dir
    )

    # 9. Compute RMSE improvement by lead time.
    _evaluate_by_lead_time(
        ob, ec, pdfm, mlr, lgb, figure_dir
    )

    # 10. Compute RMSE improvement by forecast hour.
    _evaluate_by_forecast_hour(
        ob, ec, pdfm, mlr, lgb, forecast_hour_ind, figure_dir
    )

    # 11. Load station metadata and compute RMSE improvement per station.
    rmse_improve_sta, station_info = _evaluate_by_station(
        ob, ec, pdfm, mlr, lgb, station_csv, figure_dir,
        cfg['constants']['station_num']
    )

    # 12. Plot RMSE improvement versus longitude.
    access97_plots.plot_rmse_scatter(
        station_info,
        rmse_improve_sta,
        figure_dir,
        var_name='经度',
        x_label='经度',
        x_lims=(108.65, 114.4),
        x_ticks=[109, 110, 111, 112, 113, 114],
        x_tick_labels=['109°', '110°', '111°', '112°', '113°', '114°E'],
        filename='rmse_improvement-lon.jpg',
    )

    # 13. Plot RMSE improvement versus latitude.
    access97_plots.plot_rmse_scatter(
        station_info,
        rmse_improve_sta,
        figure_dir,
        var_name='纬度',
        x_label='纬度',
        x_lims=(24.5, 30.25),
        x_ticks=[25, 26, 27, 28, 29, 30],
        x_tick_labels=['25°', '26°', '27°', '28°', '29°', '30°N'],
        filename='rmse_improvement-lat.jpg',
    )

    # 14. Plot RMSE improvement versus altitude.
    access97_plots.plot_rmse_scatter(
        station_info,
        rmse_improve_sta,
        figure_dir,
        var_name='海拔',
        x_label='高程/m',
        x_lims=(0, 1500),
        x_ticks=[0, 300, 600, 900, 1200, 1500],
        x_tick_labels=['0', '300', '600', '900', '1200', '1500'],
        filename='rmse_improvement-alti.jpg',
    )

    # 15. Plot violin distribution of RMSE improvement by model.
    access97_plots.plot_rmse_violin(rmse_improve_sta, figure_dir)


if __name__ == '__main__':
    print('Program access97.py started')
    start = arrow.now()

    main()

    end = arrow.now()
    elapsed = (end - start).total_seconds()
    print(
        f'Program access97.py finished, total time '
        f'{utils.format_time(elapsed)}'
    )
