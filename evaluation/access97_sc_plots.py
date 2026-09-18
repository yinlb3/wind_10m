# -*- coding: utf-8 -*-
"""
Plotting helpers for evaluation/access97_sc.py.

Founded in 2026-09-15
Modified in 2026-09-18
@author: yinlb, deepseek-flash
"""

import pathlib
import typing

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from meteva import base as meb

from src import metrics


ELEMENT_NAME_LIST = [
    'T_2m', 'MSL', 'D_2m', 'U_10m', 'V_10m', 'T_1000hPa',
    'GH_1000hPa', 'R_1000hPa', 'U_1000hPa', 'V_1000hPa',
    'T_925hPa', 'GH_925hPa', 'R_925hPa', 'U_925hPa', 'V_925hPa',
    'T_850hPa', 'GH_850hPa', 'R_850hPa', 'U_850hPa', 'V_850hPa',
    'T_700hPa', 'GH_700hPa', 'R_700hPa', 'U_700hPa', 'V_700hPa',
    'T_500hPa', 'GH_500hPa', 'R_500hPa', 'U_500hPa', 'V_500hPa'
]


def plot_mi_heatmap(mi: np.ndarray, fig_dir: pathlib.Path) -> None:
    """
    Plot the mutual-information heatmap.

    Args:
        mi (np.ndarray): Mutual-information matrix of shape (30, 30).
        fig_dir (pathlib.Path): Directory to save figures.
    """
    sns.heatmap(mi, cmap='Reds', vmin=0, vmax=10, linewidths=0.3)
    plt.xticks(np.arange(30), ELEMENT_NAME_LIST, rotation=45, fontsize=7)
    plt.yticks(np.arange(30) + 0.5, ELEMENT_NAME_LIST, rotation=0, fontsize=7)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.savefig(fig_dir / 'mi.jpg', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close()
    sns.heatmap(mi, cmap='binary', vmin=0, vmax=10, linewidths=0.3)
    plt.xticks(np.arange(30), ELEMENT_NAME_LIST, rotation=45, fontsize=7)
    plt.yticks(np.arange(30) + 0.5, ELEMENT_NAME_LIST, rotation=0, fontsize=7)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.savefig(fig_dir / 'mi2.jpg', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close()


def plot_grade_bars(
    ts_grade: np.ndarray, wd_kappa_grade: np.ndarray, fig_dir: pathlib.Path
) -> None:
    """
    Plot wind-speed grade and wind-direction kappa bar charts.

    Args:
        ts_grade (np.ndarray): Threat-score grades of shape (2, 4, 8).
        wd_kappa_grade (np.ndarray): Wind-direction kappa of shape (4, 8).
        fig_dir (pathlib.Path): Directory to save figures.
    """
    # 3. Plot wind-speed grade and wind-direction kappa bar charts.
    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(8) - 0.3,
        height=ts_grade[0, 0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(8) - 0.1,
        height=ts_grade[0, 1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(8) + 0.1,
        height=ts_grade[0, 2, :],
        width=0.2,
        color='yellow',
        label='MOS'
    )
    plt.bar(
        x=np.arange(8) + 0.3,
        height=ts_grade[0, 3, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 8))
    plt.xticks(range(8), ['1级', '2级', '3级', '4级', '5级', '6级', '7级', '8级及以上'])
    plt.ylim((0, 1))
    plt.yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
    plt.xlabel('风速等级')
    plt.ylabel('TS')
    plt.legend(loc='upper right')
    plt.savefig(fig_dir / 'ts_grade.jpg', bbox_inches='tight', dpi=800)
    plt.close()

    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(8) - 0.3,
        height=ts_grade[1, 0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(8) - 0.1,
        height=ts_grade[1, 1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(8) + 0.1,
        height=ts_grade[1, 2, :],
        width=0.2,
        color='yellow',
        label='MOS'
    )
    plt.bar(
        x=np.arange(8) + 0.3,
        height=ts_grade[1, 3, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 8))
    thresholds = [
        '≥0.3m/s', '≥1.6m/s', '≥3.4m/s', '≥5.5m/s',
        '≥8.0m/s', '≥10.8m/s', '≥13.9m/s', '≥17.2m/s'
    ]
    plt.xticks(range(8), thresholds)
    plt.ylim((0, 1))
    plt.yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
    plt.xlabel('风速等级')
    plt.ylabel('TS')
    plt.legend(loc='upper right')
    plt.savefig(fig_dir / 'ts_grade+.jpg', bbox_inches='tight', dpi=800)
    plt.close()

    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(8) - 0.3,
        height=wd_kappa_grade[0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(8) - 0.1,
        height=wd_kappa_grade[1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(8) + 0.1,
        height=wd_kappa_grade[2, :],
        width=0.2,
        color='yellow',
        label='MOS'
    )
    plt.bar(
        x=np.arange(8) + 0.3,
        height=wd_kappa_grade[3, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 8))
    plt.xticks(range(8), ['1级', '2级', '3级', '4级', '5级', '6级', '7级', '8级及以上'])
    plt.ylim((0, 1))
    plt.yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
    plt.xlabel('风速等级')
    plt.ylabel('kappa')
    plt.legend(loc='upper left')
    plt.savefig(fig_dir / 'wd_kappa_grade.jpg', bbox_inches='tight', dpi=800)
    plt.close()


def plot_yearly_rmse(
    ws_rmse_year: np.ndarray, wd_rmse_year: np.ndarray, fig_dir: pathlib.Path
) -> None:
    """
    Plot yearly RMSE bar charts.

    Args:
        ws_rmse_year (np.ndarray): Wind-speed RMSE of shape (4, 6).
        wd_rmse_year (np.ndarray): Wind-direction RMSE of shape (4, 6).
        fig_dir (pathlib.Path): Directory to save figures.
    """
    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(6) - 0.3,
        height=ws_rmse_year[0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(6) - 0.1,
        height=ws_rmse_year[1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(6) + 0.1,
        height=ws_rmse_year[2, :],
        width=0.2,
        color='yellow',
        label='MOS'
    )
    plt.bar(
        x=np.arange(6) + 0.3,
        height=ws_rmse_year[3, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 6))
    plt.xticks(range(6), [f'{idx + 2017}年' for idx in range(6)])
    plt.ylim((0, 1.6))
    plt.yticks((0, 0.4, 0.8, 1.2, 1.6))
    plt.xlabel('年份')
    plt.ylabel('风速RMSE (m/s)')
    plt.legend(loc='upper center', ncols=4)
    plt.savefig(fig_dir / 'ws_rmse_year.jpg', bbox_inches='tight', dpi=800)
    plt.close()

    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(6) - 0.3,
        height=wd_rmse_year[0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(6) - 0.1,
        height=wd_rmse_year[1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(6) + 0.1,
        height=wd_rmse_year[2, :],
        width=0.2,
        color='yellow',
        label='MOS'
    )
    plt.bar(
        x=np.arange(6) + 0.3,
        height=wd_rmse_year[3, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 6))
    plt.xticks(range(6), [f'{idx + 2017}年' for idx in range(6)])
    plt.ylim((0, 100))
    plt.yticks((0, 20, 40, 60, 80, 100))
    plt.xlabel('年份')
    plt.ylabel('风向RMSE (°)')
    plt.legend(loc='upper center', ncols=4)
    plt.savefig(fig_dir / 'wd_rmse_year.jpg', bbox_inches='tight', dpi=800)
    plt.close()


def plot_monthly_rmse(
    ws_rmse_month: np.ndarray, wd_rmse_month: np.ndarray, fig_dir: pathlib.Path
) -> None:
    """
    Plot monthly RMSE bar charts.

    Args:
        ws_rmse_month (np.ndarray): Wind-speed RMSE of shape (4, 12).
        wd_rmse_month (np.ndarray): Wind-direction RMSE of shape (4, 12).
        fig_dir (pathlib.Path): Directory to save figures.
    """
    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(12) - 0.3,
        height=ws_rmse_month[0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(12) - 0.1,
        height=ws_rmse_month[1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(12) + 0.1,
        height=ws_rmse_month[2, :],
        width=0.2,
        color='yellow',
        label='MOS'
    )
    plt.bar(
        x=np.arange(12) + 0.3,
        height=ws_rmse_month[3, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 12))
    plt.xticks(range(12), [f'{idx + 1}月' for idx in range(12)])
    plt.ylim((0, 1.6))
    plt.yticks((0, 0.4, 0.8, 1.2, 1.6))
    plt.xlabel('月份')
    plt.ylabel('风速RMSE (m/s)')
    plt.legend(loc='upper center', ncols=4)
    plt.savefig(fig_dir / 'ws_rmse_month.jpg', bbox_inches='tight', dpi=800)
    plt.close()

    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(12) - 0.3,
        height=wd_rmse_month[0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(12) - 0.1,
        height=wd_rmse_month[1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(12) + 0.1,
        height=wd_rmse_month[2, :],
        width=0.2,
        color='yellow',
        label='MOS'
    )
    plt.bar(
        x=np.arange(12) + 0.3,
        height=wd_rmse_month[3, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 12))
    plt.xticks(range(12), [f'{idx + 1}月' for idx in range(12)])
    plt.ylim((0, 100))
    plt.yticks((0, 20, 40, 60, 80, 100))
    plt.xlabel('月份')
    plt.ylabel('风向RMSE (°)')
    plt.legend(loc='upper center', ncols=4)
    plt.savefig(fig_dir / 'wd_rmse_month.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    print(
        f'monthly wind speed RMSE range (all models): '
        f'{np.nanmin(ws_rmse_month):.4f} ~ {np.nanmax(ws_rmse_month):.4f} m/s'
    )
    print(
        f'monthly wind direction RMSE range (all models): '
        f'{np.nanmin(wd_rmse_month):.2f} ~ {np.nanmax(wd_rmse_month):.2f} °'
    )


def plot_leadtime_rmse(
    ws_rmse_vt: np.ndarray, wd_rmse_vt: np.ndarray, fig_dir: pathlib.Path
) -> None:
    """
    Plot forecast-lead-time RMSE bar charts.

    Args:
        ws_rmse_vt (np.ndarray): Wind-speed RMSE of shape (4, 24).
        wd_rmse_vt (np.ndarray): Wind-direction RMSE of shape (4, 24).
        fig_dir (pathlib.Path): Directory to save figures.
    """
    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(24) - 0.3,
        height=ws_rmse_vt[0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(24) - 0.1,
        height=ws_rmse_vt[1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(24) + 0.1,
        height=ws_rmse_vt[2, :],
        width=0.2,
        color='yellow',
        label='MOS'
    )
    plt.bar(
        x=np.arange(24) + 0.3,
        height=ws_rmse_vt[3, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 24))
    plt.xticks(range(24), [str(idx + 1) for idx in range(24)])
    plt.ylim((0, 1.6))
    plt.yticks((0, 0.4, 0.8, 1.2, 1.6))
    plt.xlabel('预报时效 (h)')
    plt.ylabel('风速RMSE (m/s)')
    plt.legend(loc='upper center', ncols=4)
    plt.savefig(fig_dir / 'ws_rmse_vt.jpg', bbox_inches='tight', dpi=800)
    plt.close()

    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(24) - 0.3,
        height=wd_rmse_vt[0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(24) - 0.1,
        height=wd_rmse_vt[1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(24) + 0.1,
        height=wd_rmse_vt[2, :],
        width=0.2,
        color='yellow',
        label='MOS'
    )
    plt.bar(
        x=np.arange(24) + 0.3,
        height=wd_rmse_vt[3, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 24))
    plt.xticks(range(24), [str(idx + 1) for idx in range(24)])
    plt.ylim((0, 100))
    plt.yticks((0, 20, 40, 60, 80, 100))
    plt.xlabel('预报时效 (h)')
    plt.ylabel('风向RMSE (°)')
    plt.legend(loc='upper center', ncols=4)
    plt.savefig(fig_dir / 'wd_rmse_vt.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    print(
        f'lead-time wind speed RMSE range (all models): '
        f'{np.nanmin(ws_rmse_vt):.4f} ~ {np.nanmax(ws_rmse_vt):.4f} m/s'
    )
    print(
        f'lead-time wind direction RMSE range (all models): '
        f'{np.nanmin(wd_rmse_vt):.2f} ~ {np.nanmax(wd_rmse_vt):.2f} °'
    )


def plot_forecast_hour_rmse(
    ws_rmse_fhour: np.ndarray, wd_rmse_fhour: np.ndarray, fig_dir: pathlib.Path
) -> None:
    """
    Plot forecast-hour-of-day RMSE bar charts.

    Args:
        ws_rmse_fhour (np.ndarray): Wind-speed RMSE of shape (4, 24).
        wd_rmse_fhour (np.ndarray): Wind-direction RMSE of shape (4, 24).
        fig_dir (pathlib.Path): Directory to save figures.
    """
    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(24) - 0.3,
        height=ws_rmse_fhour[0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(24) - 0.1,
        height=ws_rmse_fhour[1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(24) + 0.1,
        height=ws_rmse_fhour[2, :],
        width=0.2,
        color='yellow',
        label='MOS'
    )
    plt.bar(
        x=np.arange(24) + 0.3,
        height=ws_rmse_fhour[3, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 24))
    plt.xticks(range(24), [f'{idx}:00' for idx in range(24)])
    plt.ylim((0, 1.6))
    plt.yticks((0, 0.4, 0.8, 1.2, 1.6))
    plt.xlabel('预报时间 (UTC)')
    plt.ylabel('风速RMSE (m/s)')
    plt.legend(loc='upper center', ncols=4)
    plt.savefig(fig_dir / 'ws_rmse_fhour.jpg', bbox_inches='tight', dpi=800)
    plt.close()

    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(24) - 0.3,
        height=wd_rmse_fhour[0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(24) - 0.1,
        height=wd_rmse_fhour[1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(24) + 0.1,
        height=wd_rmse_fhour[2, :],
        width=0.2,
        color='yellow',
        label='MOS'
    )
    plt.bar(
        x=np.arange(24) + 0.3,
        height=wd_rmse_fhour[3, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 24))
    plt.xticks(range(24), [f'{idx}:00' for idx in range(24)])
    plt.ylim((0, 100))
    plt.yticks((0, 20, 40, 60, 80, 100))
    plt.xlabel('预报时间 (UTC)')
    plt.ylabel('风向RMSE (°)')
    plt.legend(loc='upper center', ncols=4)
    plt.savefig(fig_dir / 'wd_rmse_fhour.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    print(
        f'forecast-hour wind speed RMSE range (all models): '
        f'{np.nanmin(ws_rmse_fhour):.4f} ~ {np.nanmax(ws_rmse_fhour):.4f} m/s'
    )
    print(
        f'forecast-hour wind direction RMSE range (all models): '
        f'{np.nanmin(wd_rmse_fhour):.2f} ~ {np.nanmax(wd_rmse_fhour):.2f} °'
    )


def analyze_stations(
    ob2: np.ndarray, ec2: np.ndarray, pdfm2: np.ndarray,
    mlr2: np.ndarray, lgb2: np.ndarray,
    station_csv: pathlib.Path, fig_dir: pathlib.Path,
) -> None:
    """
    Compute station-level RMSE and draw violin plots.

    Args:
        ob2 (np.ndarray): Observation array of shape (n_time, n_hour, n_sta).
        ec2 (np.ndarray): ECMWF-IFS forecast array.
        pdfm2 (np.ndarray): PDFM forecast array.
        mlr2 (np.ndarray): MLR forecast array.
        lgb2 (np.ndarray): LightGBM forecast array.
        station_csv (pathlib.Path): Path to the station metadata CSV.
        fig_dir (pathlib.Path): Directory to save figures.
    """
    # 5. Station-based analysis and violin plots.
    station_information = pd.read_csv(
        station_csv, encoding='gb2312', low_memory=False
    )
    station_information.sort_values(by=['台站号'], inplace=True)
    station_information.reset_index(drop=True, inplace=True)
    sta = station_information.loc[:, ('台站号', '经度', '纬度')]
    sta = meb.sta_data(sta, columns=['id', 'lon', 'lat'])
    sta.loc[:, 'level'] = 0
    sta = sta.astype({'time': 'datetime64[ns]'})
    sta.loc[:, 'time'] = '2024-01-01 08:00:00'
    sta.loc[:, 'dtime'] = 0
    n_stations = station_information.shape[0]
    ws_rmse_sta = np.zeros((4, n_stations), dtype=np.float32) + np.nan
    wd_rmse_sta = np.zeros((4, n_stations), dtype=np.float32) + np.nan
    ws_rmse_impr_sta = np.zeros((3, n_stations), dtype=np.float32) + np.nan
    wd_rmse_impr_sta = np.zeros((3, n_stations), dtype=np.float32) + np.nan
    for i in range(n_stations):
        acc = metrics.Acc(ob2[:, :, i], ec2[:, :, i])
        ws_rmse_sta[0, i] = acc.get_ws_rmse()
        wd_rmse_sta[0, i] = acc.get_wd_rmse()
        acc = metrics.Acc(ob2[:, :, i], pdfm2[:, :, i])
        ws_rmse_sta[1, i] = acc.get_ws_rmse()
        wd_rmse_sta[1, i] = acc.get_wd_rmse()
        ws_rmse_impr_sta[0, i] = (
            (ws_rmse_sta[0, i] - ws_rmse_sta[1, i])
            / ws_rmse_sta[0, i] * 100
        )
        wd_rmse_impr_sta[0, i] = (
            (wd_rmse_sta[0, i] - wd_rmse_sta[1, i])
            / wd_rmse_sta[0, i] * 100
        )
        acc = metrics.Acc(ob2[:, :, i], mlr2[:, :, i])
        ws_rmse_sta[2, i] = acc.get_ws_rmse()
        wd_rmse_sta[2, i] = acc.get_wd_rmse()
        ws_rmse_impr_sta[1, i] = (
            (ws_rmse_sta[0, i] - ws_rmse_sta[2, i])
            / ws_rmse_sta[0, i] * 100
        )
        wd_rmse_impr_sta[1, i] = (
            (wd_rmse_sta[0, i] - wd_rmse_sta[2, i])
            / wd_rmse_sta[0, i] * 100
        )
        acc = metrics.Acc(ob2[:, :, i], lgb2[:, :, i])
        ws_rmse_sta[3, i] = acc.get_ws_rmse()
        wd_rmse_sta[3, i] = acc.get_wd_rmse()
        ws_rmse_impr_sta[2, i] = (
            (ws_rmse_sta[0, i] - ws_rmse_sta[3, i])
            / ws_rmse_sta[0, i] * 100
        )
        wd_rmse_impr_sta[2, i] = (
            (wd_rmse_sta[0, i] - wd_rmse_sta[3, i])
            / wd_rmse_sta[0, i] * 100
        )

    plt.figure(figsize=(5, 5))
    sns.violinplot(
        data={
            'ECMWF-IFS': ws_rmse_sta[0, :],
            'PDFM': ws_rmse_sta[1, :],
            'MOS': ws_rmse_sta[2, :],
            'LightGBM': ws_rmse_sta[3, :]
        },
        palette=['blue', 'green', 'yellow', 'red']
    )
    plt.xlabel('模型')
    plt.ylabel('风速RMSE (m/s)')
    plt.savefig(fig_dir / 'ws_rmse_boxplot.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    print(
        'station-wise median wind speed RMSE (m/s) '
        '[ECMWF-IFS, PDFM, MOS, LightGBM]:'
    )
    print(np.round(np.median(ws_rmse_sta, axis=1), 4))

    plt.figure(figsize=(5, 5))
    sns.violinplot(
        data={
            'ECMWF-IFS': wd_rmse_sta[0, :],
            'PDFM': wd_rmse_sta[0, :],
            'MOS': wd_rmse_sta[1, :],
            'LightGBM': wd_rmse_sta[2, :]
        },
        palette=['blue', 'green', 'yellow', 'red']
    )
    plt.xlabel('模型')
    plt.ylabel('风向RMSE (°)')
    plt.savefig(fig_dir / 'wd_rmse_boxplot.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    print(
        'station-wise median wind direction RMSE (deg) '
        '[ECMWF-IFS, PDFM, MOS, LightGBM]:'
    )
    print(np.round(np.median(wd_rmse_sta, axis=1), 2))

    plt.figure(figsize=(5, 5))
    sns.violinplot(
        data={
            'PDFM': ws_rmse_impr_sta[0, :],
            'MOS': ws_rmse_impr_sta[1, :],
            'LightGBM': ws_rmse_impr_sta[2, :]
        },
        palette=['green', 'yellow', 'red']
    )
    plt.xlabel('模型')
    plt.ylabel('风速RMSE改善率 (%)')
    plt.savefig(
        fig_dir / 'ws_rmse_improvement_boxplot.jpg',
        bbox_inches='tight', dpi=800
    )
    plt.close()
    print(
        'station-wise median wind speed RMSE improvement (%) '
        '[PDFM, MOS, LightGBM]:'
    )
    print(np.round(np.median(ws_rmse_impr_sta, axis=1), 2))

    plt.figure(figsize=(5, 5))
    sns.violinplot(
        data={
            'PDFM': wd_rmse_impr_sta[0, :],
            'MOS': wd_rmse_impr_sta[1, :],
            'LightGBM': wd_rmse_impr_sta[2, :]
        },
        palette=['green', 'yellow', 'red']
    )
    plt.xlabel('模型')
    plt.ylabel('风向RMSE改善率 (%)')
    plt.savefig(
        fig_dir / 'wd_rmse_improvement_boxplot.jpg',
        bbox_inches='tight', dpi=800
    )
    plt.close()
    print(
        'station-wise median wind direction RMSE improvement (%) '
        '[PDFM, MOS, LightGBM]:'
    )
    print(np.round(np.median(wd_rmse_impr_sta, axis=1), 2))


def plot_radar(
    ws_rmse_wd: np.ndarray,
    wd_rmse_wd: np.ndarray,
    fig_dir: pathlib.Path,
) -> None:
    """
    Plot radar charts for RMSE by wind direction.

    Args:
        ws_rmse_wd (np.ndarray): Wind-speed RMSE of shape (4, 9).
        wd_rmse_wd (np.ndarray): Wind-direction RMSE of shape (4, 9).
        fig_dir (pathlib.Path): Directory to save figures.
    """
    # 6. Radar plots by wind direction.
    ws_rmse_impr_wd = np.zeros((3, 9), dtype=np.float32) + np.nan
    wd_rmse_impr_wd = np.zeros((3, 9), dtype=np.float32) + np.nan
    ws_rmse_impr_wd[0, :] = (
        (ws_rmse_wd[0, :] - ws_rmse_wd[1, :]) / ws_rmse_wd[0, :] * 100
    )
    ws_rmse_impr_wd[1, :] = (
        (ws_rmse_wd[0, :] - ws_rmse_wd[2, :]) / ws_rmse_wd[0, :] * 100
    )
    ws_rmse_impr_wd[2, :] = (
        (ws_rmse_wd[0, :] - ws_rmse_wd[3, :]) / ws_rmse_wd[0, :] * 100
    )
    wd_rmse_impr_wd[0, :] = (
        (wd_rmse_wd[0, :] - wd_rmse_wd[1, :]) / wd_rmse_wd[0, :] * 100
    )
    wd_rmse_impr_wd[1, :] = (
        (wd_rmse_wd[0, :] - wd_rmse_wd[2, :]) / wd_rmse_wd[0, :] * 100
    )
    wd_rmse_impr_wd[2, :] = (
        (wd_rmse_wd[0, :] - wd_rmse_wd[3, :]) / wd_rmse_wd[0, :] * 100
    )
    labels = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
    angles = np.linspace(0, 2 * np.pi, 9).tolist()

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, ws_rmse_wd[0, :], color='blue', label='ECMWF-IFS')
    ax.plot(angles, ws_rmse_wd[1, :], color='green', label='PDFM')
    ax.plot(angles, ws_rmse_wd[2, :], color='yellow', label='MOS')
    ax.plot(angles, ws_rmse_wd[3, :], color='red', label='LightGBM')
    ax.set_xticks(angles[:-1], labels)
    ax.set_rgrids((0, 0.5, 1, 1.5, 2), angle=90)
    plt.savefig(fig_dir / 'radar_ws_rmse.jpg', bbox_inches='tight', dpi=800)
    print(
        f'max wind speed RMSE by wind direction: '
        f'{np.nanmax(ws_rmse_wd):.4f} m/s'
    )
    plt.close()

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, wd_rmse_wd[0, :], color='blue', label='ECMWF-IFS')
    ax.plot(angles, wd_rmse_wd[1, :], color='green', label='PDFM')
    ax.plot(angles, wd_rmse_wd[2, :], color='yellow', label='MOS')
    ax.plot(angles, wd_rmse_wd[3, :], color='red', label='LightGBM')
    ax.set_xticks(angles[:-1], labels)
    ax.set_rgrids((0, 25, 50, 75, 100), angle=90)
    plt.savefig(fig_dir / 'radar_wd_rmse.jpg', bbox_inches='tight', dpi=800)
    print(
        f'max wind direction RMSE by wind direction: '
        f'{np.nanmax(wd_rmse_wd):.2f} °'
    )
    plt.close()

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, ws_rmse_impr_wd[0, :], color='green', label='PDFM')
    ax.plot(angles, ws_rmse_impr_wd[1, :], color='yellow', label='MOS')
    ax.plot(angles, ws_rmse_impr_wd[2, :], color='red', label='LightGBM')
    ax.set_xticks(angles[:-1], labels)
    ax.set_rgrids((0, 10, 20, 30, 40, 50), angle=90)
    plt.savefig(
        fig_dir / 'radar_ws_improvement_rmse.jpg',
        bbox_inches='tight', dpi=800
    )
    print(
        f'wind speed RMSE improvement by wind direction: '
        f'{np.nanmin(ws_rmse_impr_wd):.2f}% ~ {np.nanmax(ws_rmse_impr_wd):.2f}%'
    )
    plt.close()

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, wd_rmse_impr_wd[0, :], color='green', label='PDFM')
    ax.plot(angles, wd_rmse_impr_wd[1, :], color='yellow', label='MOS')
    ax.plot(angles, wd_rmse_impr_wd[2, :], color='red', label='LightGBM')
    ax.set_xticks(angles[:-1], labels)
    ax.set_rgrids((-10, -5, 0, 5, 10, 15), angle=90)
    plt.savefig(
        fig_dir / 'radar_wd_improvement_rmse.jpg',
        bbox_inches='tight', dpi=800
    )
    print(
        f'wind direction RMSE improvement by wind direction: '
        f'{np.nanmin(wd_rmse_impr_wd):.2f}% ~ {np.nanmax(wd_rmse_impr_wd):.2f}%'
    )
    plt.close()


def compare_lgb_variants(
    ob2: np.ndarray,
    path_97sc: pathlib.Path,
    years: typing.List[int],
) -> None:
    """
    Compare the four LightGBM training variants.

    Args:
        ob2 (np.ndarray): Observation array.
        path_97sc (pathlib.Path): Directory of the SC-scheme predictions.
        years (typing.List[int]): Years to concatenate.
    """
    lgb_1 = list()
    lgb_2 = list()
    lgb_3 = list()
    # lgb_4 = list()
    for year in years:
        lgb_1.append(np.load(path_97sc / f'lgb_1_{year}.npy'))
        lgb_2.append(np.load(path_97sc / f'lgb_2_{year}.npy'))
        lgb_3.append(np.load(path_97sc / f'lgb_3_{year}.npy'))
        # lgb_4.append(np.load(path_97sc / f'lgb_4_{year}.npy'))
    lgb_1 = np.vstack(lgb_1)
    lgb_2 = np.vstack(lgb_2)
    lgb_3 = np.vstack(lgb_3)
    # lgb_4 = np.vstack(lgb_4)

    variants = [
        ('LightGBM1', lgb_1),
        ('LightGBM2', lgb_2),
        ('LightGBM3', lgb_3),
        # ('LightGBM4', lgb_4),
    ]
    for name, pred in variants:
        print(f'----- {name} -----')
        acc = metrics.Acc(ob2, pred)
        print(f'  Wind speed ME    : {acc.get_ws_me():.4f} m/s')
        print(f'  Wind speed CC    : {acc.get_ws_r():.4f}')
        print(f'  Wind speed MAE   : {acc.get_ws_mae():.4f} m/s')
        print(f'  Wind speed RMSE  : {acc.get_ws_rmse():.4f} m/s')
        print(f'  Wind speed MRE   : {acc.get_ws_mre():.4f}')
        print(f'  Wind speed OA    : {acc.get_ws_oa():.4f}')
        print(f'  Wind speed kappa : {acc.get_ws_kappa():.4f}')
        print(f'  Wind dir ME      : {acc.get_wd_me():.2f} °')
        print(f'  Wind dir MAE     : {acc.get_wd_mae():.2f} °')
        print(f'  Wind dir RMSE    : {acc.get_wd_rmse():.2f} °')
        print(f'  Wind dir OA      : {acc.get_wd_oa():.4f}')
        print(f'  Wind dir kappa   : {acc.get_wd_kappa():.4f}')
