#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw the figures of the 2411-station experiment.

Reads the CSV tables written by verify_2411.py plus the mutual-information
matrix written by compute_mi_2411.py, and saves the figures to the result
directory.

Founded in 2026-10-29
Modified in 2026-10-29
@author: yinlb, deepseek-flash
"""

import pathlib
import typing

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

MODELS = ['ECMWF-IFS', 'PDFM', 'MLR', 'LightGBM-A', 'LightGBM-B']
COLORS = ['#1f77b4', '#2ca02c', '#bcbd22', '#d62728', '#9467bd']
SECTORS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
DPI = 300
# Below this station count the station-wise distribution and the high speed
# grades are not representative, so the figures are marked as provisional.
MIN_STATIONS = 50


def _read(path: pathlib.Path) -> typing.Optional[pd.DataFrame]:
    """
    Read one result table.

    Args:
        path: CSV path.

    Returns:
        typing.Optional[pandas.DataFrame]: Table or None when missing.
    """
    if not path.exists():
        print('  missing {0}'.format(path.name), flush=True)
        return None
    return pd.read_csv(path, encoding='utf-8-sig')


def plot_mi(result_dir: pathlib.Path, fig_dir: pathlib.Path) -> None:
    """
    Draw the mutual-information heat map (Figure 1).

    Args:
        result_dir: Directory holding mi_2411.npy.
        fig_dir: Output directory for figures.
    """
    path = result_dir / 'mi_2411.npy'
    names_path = result_dir / 'mi_2411_names.txt'
    if not path.exists() or not names_path.exists():
        print('  missing mi_2411.npy', flush=True)
        return
    matrix = np.load(path)
    names = names_path.read_text(encoding='utf-8').split()
    fig, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap='viridis')
    axis.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
    axis.set_yticks(range(len(names)), names, fontsize=7)
    fig.colorbar(image, ax=axis, shrink=0.8)
    axis.set_title('Mutual information of the factors')
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig1_mi.jpg', dpi=DPI)
    plt.close(fig)


def plot_ts(result_dir: pathlib.Path, fig_dir: pathlib.Path) -> None:
    """
    Draw the threat score of the speed grades (Figure 2a).

    Args:
        result_dir: Directory holding table5_overall.csv.
        fig_dir: Output directory for figures.
    """
    table = _read(result_dir / 'table5_overall.csv')
    if table is None:
        return
    columns = ['ws_ts{0}'.format(i) for i in range(1, 9)]
    x_val = np.arange(len(columns))
    width = 0.16
    fig, axis = plt.subplots(figsize=(10, 4.5))
    for index, model in enumerate(MODELS):
        row = table.loc[table['model'] == model]
        if row.empty:
            continue
        values = row[columns].to_numpy(dtype=float).ravel()
        missing = [i + 1 for i, v in enumerate(values) if np.isnan(v)]
        if missing:
            print(
                '  {0}: no samples at grade {1}'.format(model, missing),
                flush=True,
            )
        axis.bar(
            x=x_val + (index - 2) * width, height=values, width=width,
            color=COLORS[index], label=model,
        )
    axis.set_xticks(x_val, ['{0}'.format(i) for i in range(1, 9)])
    axis.set_xlabel('Wind speed grade')
    axis.set_ylabel('Threat score')
    axis.set_ylim(0, 1)
    axis.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig2a_ts_grade.jpg', dpi=DPI)
    plt.close(fig)


def _panel(
    axis: plt.Axes, table: pd.DataFrame, column: str,
    xlabel: str, ylabel: str,
) -> None:
    """
    Draw one RMSE panel.

    Args:
        axis: Target axis.
        table: Table with one column per model and an index axis.
        column: Column prefix, either 'ws' or 'wd'.
        xlabel: X axis label.
        ylabel: Y axis label.
    """
    for index, model in enumerate(MODELS):
        if model not in table.columns:
            continue
        axis.plot(
            np.arange(table.shape[0]), table[model].to_numpy(),
            marker='o', markersize=3, color=COLORS[index], label=model,
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.3)


def plot_rmse_panels(result_dir: pathlib.Path, fig_dir: pathlib.Path) -> None:
    """
    Draw the year/month/lead-time/hour RMSE panels (Figure 3).

    Args:
        result_dir: Directory holding the group tables.
        fig_dir: Output directory for figures.
    """
    by_year = _read(result_dir / 'table5_by_year.csv')
    groups = {
        'month': _read(result_dir / 'group_month_rmse.csv'),
        'lead': _read(result_dir / 'group_lead_rmse.csv'),
        'hour': _read(result_dir / 'group_hour_rmse.csv'),
    }
    if by_year is None or any(item is None for item in groups.values()):
        return

    year_ws = by_year.pivot(index='year', columns='model', values='ws_rmse')
    year_wd = by_year.pivot(index='year', columns='model', values='wd_rmse')
    fig, axes = plt.subplots(4, 2, figsize=(12, 13))
    _panel(axes[0, 0], year_ws, 'ws', 'Year', 'Speed RMSE (m/s)')
    _panel(axes[0, 1], year_wd, 'wd', 'Year', 'Direction RMSE (deg)')
    _panel(axes[1, 0], groups['month'].set_index('group'), 'ws',
           'Month', 'Speed RMSE (m/s)')
    _panel(axes[1, 1], groups['month'].set_index('group'), 'wd',
           'Month', 'Direction RMSE (deg)')
    _panel(axes[2, 0], groups['lead'].set_index('group'), 'ws',
           'Lead time index (13-36 h)', 'Speed RMSE (m/s)')
    _panel(axes[2, 1], groups['lead'].set_index('group'), 'wd',
           'Lead time index (13-36 h)', 'Direction RMSE (deg)')
    _panel(axes[3, 0], groups['hour'].set_index('group'), 'ws',
           'Hour of day (UTC)', 'Speed RMSE (m/s)')
    _panel(axes[3, 1], groups['hour'].set_index('group'), 'wd',
           'Hour of day (UTC)', 'Direction RMSE (deg)')
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig3_rmse_panels.jpg', dpi=DPI)
    plt.close(fig)


def plot_station(result_dir: pathlib.Path, fig_dir: pathlib.Path) -> None:
    """
    Draw the station-wise violin plots (Figure 4).

    Args:
        result_dir: Directory holding the station tables.
        fig_dir: Output directory for figures.
    """
    ws = _read(result_dir / 'group_station_ws_rmse.csv')
    wd = _read(result_dir / 'group_station_wd_rmse.csv')
    if ws is None or wd is None:
        return
    n_station = ws.shape[0]
    if n_station < MIN_STATIONS:
        stale = fig_dir / 'fig4_station_violin.jpg'
        if stale.exists():
            stale.unlink()
        print(
            '  skipped: only {0} stations, need >= {1} for a violin plot '
            '(stale figure removed)'.format(n_station, MIN_STATIONS),
            flush=True,
        )
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, table, label in (
        (axes[0, 0], ws, 'Speed RMSE (m/s)'),
        (axes[0, 1], wd, 'Direction RMSE (deg)'),
    ):
        data = [table[m].dropna().to_numpy() for m in MODELS if m in table]
        names = [m for m in MODELS if m in table]
        axis.violinplot(data, showmedians=True)
        axis.set_xticks(range(1, len(names) + 1), names, rotation=20)
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)

    base_ws = ws['ECMWF-IFS'].to_numpy(dtype=float)
    base_wd = wd['ECMWF-IFS'].to_numpy(dtype=float)
    for axis, table, base, label in (
        (axes[1, 0], ws, base_ws, 'Speed RMSE improvement (%)'),
        (axes[1, 1], wd, base_wd, 'Direction RMSE improvement (%)'),
    ):
        data = list()
        names = list()
        for model in MODELS[1:]:
            if model not in table:
                continue
            value = (base - table[model].to_numpy(dtype=float)) / base * 100.0
            data.append(value[~np.isnan(value)])
            names.append(model)
        axis.violinplot(data, showmedians=True)
        axis.axhline(0, color='k', linewidth=0.8)
        axis.set_xticks(range(1, len(names) + 1), names, rotation=20)
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / 'fig4_station_violin.jpg', dpi=DPI)
    plt.close(fig)


def plot_radar(result_dir: pathlib.Path, fig_dir: pathlib.Path) -> None:
    """
    Draw the wind-direction radar plots (Figure 5).

    Args:
        result_dir: Directory holding the sector tables.
        fig_dir: Output directory for figures.
    """
    speed = _read(result_dir / 'group_sector_speed_rmse.csv')
    direction = _read(result_dir / 'group_sector_dir_rmse.csv')
    if speed is None or direction is None:
        return

    angle = np.deg2rad(np.arange(8) * 45.0)
    closed = np.append(angle, angle[0])
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5.5), subplot_kw={'projection': 'polar'}
    )
    for axis, table, label in (
        (axes[0], speed, 'Speed RMSE (m/s)'),
        (axes[1], direction, 'Direction RMSE (deg)'),
    ):
        for index, model in enumerate(MODELS):
            if model not in table:
                continue
            values = table[model].to_numpy(dtype=float)
            axis.plot(
                closed, np.append(values, values[0]),
                color=COLORS[index], label=model,
            )
        axis.set_xticks(angle, SECTORS)
        axis.set_title(label)
        axis.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1),
                    fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig5_radar.jpg', dpi=DPI)
    plt.close(fig)


def main() -> None:
    """Draw all available figures."""
    from src import data_2411

    cfg = data_2411.load_config()
    result_dir = pathlib.Path(cfg['paths']['result_dir'])
    fig_dir = result_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    station_table = _read(result_dir / 'group_station_ws_rmse.csv')
    n_station = 0 if station_table is None else station_table.shape[0]
    if n_station < MIN_STATIONS:
        print(
            'WARNING: only {0} stations; figures are provisional - the '
            'high speed grades and the station distribution are not '
            'representative until the full run finishes'.format(n_station),
            flush=True,
        )

    for func in (
        plot_mi, plot_ts, plot_rmse_panels, plot_station, plot_radar
    ):
        print('  drawing {0}'.format(func.__name__), flush=True)
        func(result_dir, fig_dir)
    print('Figures: {0}'.format(fig_dir), flush=True)


if __name__ == '__main__':
    print('Program plot_2411.py started', flush=True)
    main()

