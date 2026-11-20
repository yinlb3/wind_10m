#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw the figures of the 2411-station experiment.

Reads the CSV tables written by verify_2411.py plus the mutual-information
matrix written by compute_mi_2411.py, and saves the figures to the result
directory.

Founded in 2026-10-29
Modified in 2026-11-20
@author: yinlb
"""

import argparse
import pathlib
import typing

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from src import data_2411  # noqa: E402

MODELS = ['ECMWF-IFS', 'PDFM', 'MLR', 'LightGBM-A', 'LightGBM-B']
# Colours follow the paper; the LightGBM-B variant of this reproduction is
# drawn in purple because the paper has no second LightGBM branch.
COLORS = ['blue', 'green', 'yellow', 'red', 'purple']
SECTORS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
SPEED_LABELS = [
    '≥0.3 m·s$^{-1}$', '≥1.6 m·s$^{-1}$', '≥3.4 m·s$^{-1}$',
    '≥5.5 m·s$^{-1}$', '≥8.0 m·s$^{-1}$', '≥10.8 m·s$^{-1}$',
    '≥13.9 m·s$^{-1}$', '≥17.2 m·s$^{-1}$',
]
GRADE_LABELS = [
    '1级', '2级', '3级', '4级', '5级', '6级', '7级', '8级及以上',
]
# The paper orders the factors by level; the cached matrix is ordered by
# variable group, so it has to be re-sorted before plotting.
FACTOR_ORDER = ['2T', 'MSL', '2D', '10U', '10V', 'PRE'] + [
    '{0}_{1}'.format(var, level)
    for level in (1000, 925, 850, 700)
    for var in ('T', 'GH', 'RH', 'U', 'V')
]
FACTOR_TAGS = {'T': 'T', 'GH': 'GH', 'RH': 'R', 'U': 'U', 'V': 'V'}
FACTOR_LABELS = {
    '2T': 'T_2m', 'MSL': 'MSL', '2D': 'D_2m', '10U': 'U_10m',
    '10V': 'V_10m', 'PRE': 'PRE',
}
for _level in (1000, 925, 850, 700):
    for _var, _tag in FACTOR_TAGS.items():
        FACTOR_LABELS['{0}_{1}'.format(_var, _level)] = (
            '{0}_{1}hPa'.format(_tag, _level)
        )

plt.rcParams['font.sans-serif'] = [
    'SimHei', 'Microsoft YaHei', 'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False
_CFG = data_2411.load_config()
FIG_DPI = int(_CFG['figure']['dpi'])
FIG_FORMATS = [
    str(item).lstrip('.').lower() for item in _CFG['figure']['formats']
]
if not FIG_FORMATS:
    raise ValueError('figure.formats is empty in config/config.2411.yaml')
# Figure widths follow the journal layout: single column 7.8 cm, double 15 cm.
CM = 1.0 / 2.54
W_SINGLE = 7.8 * CM
W_DOUBLE = 15.0 * CM

# Below this station count the station-wise distribution and the high speed
# grades are not representative, so the figures are marked as provisional.
MIN_STATIONS = 50
# Below this sample count the kappa of one speed grade is too noisy to plot.
MIN_GRADE_SAMPLES = 30


def save_fig(
    fig: plt.Figure,
    fig_dir: pathlib.Path,
    name: str,
    **kwargs: typing.Any,
) -> None:
    """
    Save one figure once per configured format.

    Args:
        fig: Figure to save.
        fig_dir: Output directory of the figures.
        name: File stem without the extension.
        kwargs: Extra arguments passed to ``savefig``.
    """
    for ext in FIG_FORMATS:
        path = fig_dir / '{0}.{1}'.format(name, ext)
        fig.savefig(path, dpi=FIG_DPI, **kwargs)


def label_panels(axes: np.ndarray, tags: str) -> None:
    """
    Add self-contained panel labels to a composite figure.

    Args:
        axes: Array of matplotlib axes.
        tags: One letter per panel, in row-major order.
    """
    for axis, tag in zip(axes.flat, tags):
        axis.text(
            -0.06, 1.06, '({0})'.format(tag), transform=axis.transAxes,
            ha='left', va='top', fontsize=9,
        )


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

    The factors are re-sorted into the level-by-level order used by the
    paper before the matrix is drawn.

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
    index = [names.index(name) for name in FACTOR_ORDER if name in names]
    matrix = matrix[np.ix_(index, index)]
    labels = [FACTOR_LABELS[names[item]] for item in index]

    fig, axis = plt.subplots(figsize=(W_DOUBLE, W_DOUBLE * 9.0 / 11.0))
    image = axis.imshow(matrix, cmap='Reds', vmin=0)
    ticks = np.arange(len(labels))
    axis.set_xticks(
        ticks, labels, rotation=45, ha='right', rotation_mode='anchor',
        fontsize=7,
    )
    axis.set_yticks(ticks, labels, fontsize=7)
    axis.set_xticks(ticks - 0.5, minor=True)
    axis.set_yticks(ticks - 0.5, minor=True)
    axis.grid(which='minor', color='white', linewidth=0.8)
    axis.tick_params(which='minor', length=0)
    fig.colorbar(image, ax=axis, shrink=0.75)
    fig.tight_layout()
    save_fig(fig, fig_dir, 'fig1_mi')
    plt.close(fig)


def _grouped_bars(
    axis: plt.Axes, table: pd.DataFrame, ticks: typing.Sequence[str],
    ylabel: str, xlabel: str, ylim: typing.Optional[float] = None,
    legend: str = 'top',
) -> None:
    """
    Draw one grouped bar chart with one group per row of the table.

    Args:
        axis: Target axis.
        table: Table with one column per model.
        ticks: Tick labels, one per row.
        ylabel: Y axis label.
        xlabel: X axis label.
        ylim: Upper limit of the y axis, auto when None.
        legend: 'top' places the legend above the axes, 'inside' keeps it
            inside the upper right corner as in the paper's Figure 2.
    """
    names = [model for model in MODELS if model in table.columns]
    width = 0.8 / max(len(names), 1)
    x_val = np.arange(table.shape[0])
    for index, model in enumerate(names):
        offset = (index - (len(names) - 1) / 2.0) * width
        axis.bar(
            x_val + offset, table[model].to_numpy(dtype=float),
            width=width, color=COLORS[MODELS.index(model)], label=model,
        )
    axis.set_xticks(x_val, ticks)
    axis.set_xlim(-0.55, table.shape[0] - 0.45)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if ylim is not None:
        axis.set_ylim(0, ylim)
    if legend == 'inside':
        axis.legend(fontsize=8, loc='upper right')
    else:
        axis.legend(
            loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(names),
            fontsize=7, frameon=True, borderaxespad=0.2,
        )


def plot_ts(result_dir: pathlib.Path, fig_dir: pathlib.Path) -> None:
    """
    Draw cumulative threat scores at the wind-speed thresholds (Figure 2a).

    Args:
        result_dir: Directory holding table5_overall.csv.
        fig_dir: Output directory for figures.
    """
    table = _read(result_dir / 'table5_overall.csv')
    if table is None:
        return
    columns = ['ws_ts{0}'.format(i) for i in range(1, 9)]
    table = table.set_index('model')
    missing = [column for column in columns if column not in table.columns]
    if missing:
        print('  missing columns {0}'.format(missing), flush=True)
        return
    values = table[columns].T
    values.index = np.arange(values.shape[0])
    for model in MODELS:
        if model not in values.columns:
            continue
        absent = [
            i + 1 for i, item in enumerate(values[model]) if np.isnan(item)
        ]
        if absent:
            print(
                '  {0}: no samples at grade {1}'.format(model, absent),
                flush=True,
            )

    fig, axis = plt.subplots(figsize=(W_SINGLE, W_SINGLE * 3.8 / 9.0))
    _grouped_bars(
        axis, values, SPEED_LABELS, 'TS', '风速阈值（m·s$^{-1}$）', ylim=1.0,
        legend='inside',
    )
    fig.tight_layout()
    save_fig(fig, fig_dir, 'fig2a_ts_grade')
    plt.close(fig)


def plot_kappa(result_dir: pathlib.Path, fig_dir: pathlib.Path) -> None:
    """
    Draw the direction kappa of the speed grades (Figure 2b).

    Args:
        result_dir: Directory holding group_grade_wd_kappa.csv.
        fig_dir: Output directory for figures.
    """
    table = _read(result_dir / 'group_grade_wd_kappa.csv')
    if table is None:
        return
    table = table.set_index('grade')
    count = _read(result_dir / 'group_grade_wd_count.csv')
    if count is not None:
        count = count.set_index('grade').reindex(table.index)
        weak = (count < MIN_GRADE_SAMPLES).reindex(
            columns=table.columns, fill_value=True
        )
        table = table.mask(weak)
        for grade in weak.index:
            if weak.loc[grade].all():
                print(
                    '  grade {0}: dropped, fewer than {1} samples'.format(
                        grade, MIN_GRADE_SAMPLES
                    ),
                    flush=True,
                )
    ticks = GRADE_LABELS[:table.shape[0]]
    fig, axis = plt.subplots(figsize=(W_SINGLE, W_SINGLE * 3.8 / 9.0))
    _grouped_bars(
        axis, table, ticks, 'kappa', '风速等级', ylim=1.0, legend='inside',
    )
    axis.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    save_fig(fig, fig_dir, 'fig2b_kappa_grade')
    plt.close(fig)


def _panel(
    axis: plt.Axes, table: pd.DataFrame, ticks: typing.Sequence[str],
    xlabel: str, ylabel: str, ylim: typing.Optional[float] = None,
) -> None:
    """
    Draw one grouped-bar RMSE panel.

    Args:
        axis: Target axis.
        table: Table with one column per model and one row per group.
        ticks: Tick labels, one per row.
        xlabel: X axis label.
        ylabel: Y axis label.
        ylim: Upper limit of the y axis, auto when None.
    """
    _grouped_bars(
        axis, table, ticks, ylabel, xlabel, ylim=ylim, legend='inside'
    )


def plot_rmse_panels(result_dir: pathlib.Path, fig_dir: pathlib.Path) -> None:
    """
    Draw the year/month/lead-time/hour RMSE panels (Figure 3).

    Args:
        result_dir: Directory holding the group tables.
        fig_dir: Output directory for figures.
    """
    paths = (
        'table5_by_year.csv',
        'group_month_rmse.csv', 'group_month_dir_rmse.csv',
    )
    tables = [_read(result_dir / name) for name in paths]
    if any(item is None for item in tables):
        return
    by_year = tables[0]
    month = tables[1].set_index('group')
    month_dir = tables[2].set_index('group')

    year_ws = by_year.pivot(index='year', columns='model', values='ws_rmse')
    year_wd = by_year.pivot(index='year', columns='model', values='wd_rmse')
    year_ticks = ['{0}年'.format(int(item)) for item in year_ws.index]
    month_ticks = ['{0}月'.format(i) for i in range(1, month.shape[0] + 1)]

    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, W_DOUBLE * 0.78))
    fig.subplots_adjust(hspace=0.55, wspace=0.28)
    _panel(axes[0, 0], year_ws, year_ticks, '年份', '风速 RMSE（m·s$^{-1}$）',
           1.6)
    _panel(axes[0, 1], year_wd, year_ticks, '年份', '风向 RMSE（°）')
    _panel(axes[1, 0], month, month_ticks, '月份', '风速 RMSE（m·s$^{-1}$）',
           1.6)
    _panel(axes[1, 1], month_dir, month_ticks, '月份', '风向 RMSE（°）',
           100.0)
    label_panels(axes, 'abcd')
    save_fig(fig, fig_dir, 'fig3_rmse_panels', bbox_inches='tight')
    plt.close(fig)


def _line_panel(
    axis: plt.Axes,
    table: pd.DataFrame,
    x_val: np.ndarray,
    xlabel: str,
    ylabel: str,
) -> None:
    """
    Draw one RMSE panel as lines, one line per model.

    Args:
        axis: Target axis.
        table: Table with one column per model, indexed by the group value.
        x_val: X coordinates of the rows.
        xlabel: X axis label.
        ylabel: Y axis label.
    """
    for model in MODELS:
        if model not in table.columns:
            continue
        axis.plot(
            x_val, table[model].to_numpy(dtype=float),
            color=COLORS[MODELS.index(model)], marker='o', markersize=1.8,
            linewidth=0.9, label=model,
        )
    axis.set_xlabel(xlabel, fontsize=7)
    axis.set_ylabel(ylabel, fontsize=7)
    axis.tick_params(labelsize=6.5)
    axis.grid(alpha=0.3, linewidth=0.5)
    axis.legend(
        fontsize=5.5, ncol=3, loc='upper center',
        bbox_to_anchor=(0.5, 1.24), frameon=False,
    )


def plot_rmse_lines(result_dir: pathlib.Path, fig_dir: pathlib.Path) -> None:
    """
    Draw the lead-time and hour-of-day RMSE as separate line charts.

    Args:
        result_dir: Directory holding the group tables.
        fig_dir: Output directory for figures.
    """
    paths = (
        'group_lead_rmse.csv', 'group_lead_dir_rmse.csv',
        'group_hour_rmse.csv', 'group_hour_dir_rmse.csv',
    )
    tables = [_read(result_dir / name) for name in paths]
    if any(item is None for item in tables):
        return
    lead = tables[0].set_index('group')
    lead_dir = tables[1].set_index('group')
    hour = tables[2].set_index('group')
    hour_dir = tables[3].set_index('group')
    items = (
        (lead, 'fig3e_lead_speed', '预报时效（h）', '风速 RMSE（m·s$^{-1}$）'),
        (lead_dir, 'fig3f_lead_dir', '预报时效（h）', '风向 RMSE（°）'),
        (hour, 'fig3g_hour_speed', '预报时间（UTC）', '风速 RMSE（m·s$^{-1}$）'),
        (hour_dir, 'fig3h_hour_dir', '预报时间（UTC）', '风向 RMSE（°）'),
    )
    for table, name, xlabel, ylabel in items:
        fig, axis = plt.subplots(figsize=(W_SINGLE, W_SINGLE * 0.78))
        x_val = table.index.to_numpy(dtype=float)
        _line_panel(axis, table, x_val, xlabel, ylabel)
        fig.tight_layout()
        save_fig(fig, fig_dir, name)
        plt.close(fig)


def _violin(
    axis: plt.Axes, data: typing.Sequence[np.ndarray],
    names: typing.Sequence[str], label: str,
) -> None:
    """
    Draw one seaborn-style violin panel.

    The kernel density is evaluated two bandwidths beyond the data range, as
    in the paper, so each violin ends in a tail instead of being cut flat at
    the axes limits. The inner box is black with a white median line.

    Args:
        axis: Target axis.
        data: One value array per model.
        names: Model names, one per value array.
        label: Y axis label.
    """
    positions = np.arange(1, len(names) + 1)
    half_width = 0.4
    for pos, values in zip(positions, data):
        item = np.asarray(values, dtype=float)
        if item.size < 2 or np.ptp(item) == 0.0:
            item = item + np.linspace(-1e-3, 1e-3, item.size)
        kde = stats.gaussian_kde(item)
        band = float(np.sqrt(kde.covariance[0, 0]))
        grid = np.linspace(
            item.min() - 2.0 * band, item.max() + 2.0 * band, 200
        )
        width = kde(grid)
        width = width / width.max() * half_width
        axis.fill_betweenx(
            grid, pos - width, pos + width,
            facecolor=COLORS[MODELS.index(names[pos - 1])],
            edgecolor='black', linewidth=1.0,
        )
        quar1, median, quar3 = np.percentile(item, [25, 50, 75])
        inter = quar3 - quar1
        low = max(float(item.min()), quar1 - 1.5 * inter)
        high = min(float(item.max()), quar3 + 1.5 * inter)
        axis.plot([pos, pos], [low, high], color='black', linewidth=1.6)
        axis.add_patch(
            plt.Rectangle(
                (pos - 0.06, quar1), 0.12, quar3 - quar1,
                facecolor='black', edgecolor='black',
            )
        )
        axis.plot(
            [pos - 0.06, pos + 0.06], [median, median],
            color='white', linewidth=1.6,
        )
    axis.set_xticks(positions, names)
    axis.set_xlim(0.4, len(names) + 0.6)
    axis.set_xlabel('模型')
    axis.set_ylabel(label)


def plot_station(
    result_dir: pathlib.Path, fig_dir: pathlib.Path,
    allow_small: bool = False,
) -> None:
    """
    Draw the station-wise violin plots (Figure 4).

    Args:
        result_dir: Directory holding the station tables.
        fig_dir: Output directory for figures.
        allow_small: Draw even when fewer than MIN_STATIONS stations are
            available; only useful for a smoke test of the plotting chain.
    """
    ws = _read(result_dir / 'group_station_ws_rmse.csv')
    wd = _read(result_dir / 'group_station_wd_rmse.csv')
    if ws is None or wd is None:
        return
    n_station = ws.shape[0]
    if n_station < MIN_STATIONS and not allow_small:
        for ext in FIG_FORMATS:
            stale = fig_dir / 'fig4_station_violin.{0}'.format(ext)
            if stale.exists():
                stale.unlink()
        print(
            '  skipped: only {0} stations, need >= {1} for a violin plot '
            '(stale figure removed; use --allow-small to force)'.format(
                n_station, MIN_STATIONS
            ),
            flush=True,
        )
        return
    if n_station < MIN_STATIONS:
        print(
            '  WARNING: {0} stations only, the violins are not '
            'representative'.format(n_station),
            flush=True,
        )

    names = [model for model in MODELS if model in ws.columns]
    fig, axes = plt.subplots(2, 2, figsize=(W_DOUBLE, W_DOUBLE * 10.0 / 12.0))
    _violin(
        axes[0, 0], [ws[name].dropna().to_numpy() for name in names],
        names, '风速 RMSE（m·s$^{-1}$）',
    )
    _violin(
        axes[0, 1], [wd[name].dropna().to_numpy() for name in names],
        names, '风向 RMSE（°）',
    )

    base_ws = ws['ECMWF-IFS'].to_numpy(dtype=float)
    base_wd = wd['ECMWF-IFS'].to_numpy(dtype=float)
    for axis, table, base, label in (
        (axes[1, 0], ws, base_ws, '风速RMSE改善率（%）'),
        (axes[1, 1], wd, base_wd, '风向RMSE改善率（%）'),
    ):
        improved = [
            model for model in MODELS[1:] if model in table.columns
        ]
        data = [
            ((base - table[model].to_numpy(dtype=float)) / base * 100.0)
            for model in improved
        ]
        data = [item[~np.isnan(item)] for item in data]
        _violin(axis, data, improved, label)

    label_panels(axes, 'abcd')
    fig.tight_layout()
    save_fig(fig, fig_dir, 'fig4_station_violin', bbox_inches='tight')
    plt.close(fig)


def _radar(
    axis: plt.Axes, table: pd.DataFrame, title: str,
    base: typing.Optional[np.ndarray] = None,
) -> None:
    """
    Draw one eight-sector radar panel.

    Negative improvement rates are retained so that degraded sectors remain
    visible rather than being clipped to zero.

    Args:
        axis: Target polar axis.
        table: Table with one column per model and rows ordered N, NE, ...
        title: Panel title.
        base: Baseline RMSE array; when given the improvement rate is drawn.
    """
    names = [
        model for model in MODELS
        if model in table.columns
        and not (base is not None and model == 'ECMWF-IFS')
    ]
    angle = np.deg2rad(np.arange(8) * 45.0)
    closed = np.append(angle, angle[0])
    for model in names:
        values = table[model].to_numpy(dtype=float)
        if base is not None:
            values = (base - values) / base * 100.0
        values = np.nan_to_num(values, nan=0.0)
        axis.plot(
            closed, np.append(values, values[0]),
            color=COLORS[MODELS.index(model)], label=model,
        )
    axis.set_theta_zero_location('N')
    axis.set_theta_direction(-1)
    axis.set_xticks(angle, SECTORS)
    axis.set_rlabel_position(0)
    axis.set_title(title, pad=20)
    axis.grid(alpha=0.6)
    axis.legend(
        loc='upper right', bbox_to_anchor=(1.32, 1.12), fontsize=7
    )


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
    speed = speed.set_index('sector').reindex(SECTORS)
    direction = direction.set_index('sector').reindex(SECTORS)
    base_speed = speed['ECMWF-IFS'].to_numpy(dtype=float)
    base_dir = direction['ECMWF-IFS'].to_numpy(dtype=float)

    fig, axes = plt.subplots(
        2, 2, figsize=(W_DOUBLE, W_DOUBLE * 12.0 / 13.0),
        subplot_kw={'projection': 'polar'},
    )
    _radar(axes[0, 0], speed, '风速 RMSE（m·s$^{-1}$）')
    _radar(axes[0, 1], direction, '风向 RMSE（°）')
    _radar(axes[1, 0], speed, '风速 RMSE 改善率（%）', base_speed)
    _radar(axes[1, 1], direction, '风向 RMSE 改善率（%）', base_dir)
    label_panels(axes, 'abcd')
    fig.tight_layout()
    save_fig(fig, fig_dir, 'fig5_radar', bbox_inches='tight')
    plt.close(fig)


def main(allow_small: bool = False) -> None:
    """
    Draw all available figures.

    Args:
        allow_small: Draw the station violins even when fewer than
            MIN_STATIONS stations are available.
    """
    cfg = _CFG
    result_dir = data_2411.result_root(cfg)
    table_dir = result_dir / 'tables'
    fig_dir = result_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    station_table = _read(table_dir / 'group_station_ws_rmse.csv')
    n_station = 0 if station_table is None else station_table.shape[0]
    if n_station < MIN_STATIONS:
        print(
            'WARNING: only {0} stations; figures are provisional - the '
            'high speed grades and the station distribution are not '
            'representative until the full run finishes'.format(n_station),
            flush=True,
        )

    for func in (
        plot_mi, plot_ts, plot_kappa, plot_rmse_panels, plot_rmse_lines,
        plot_radar
    ):
        print('  drawing {0}'.format(func.__name__), flush=True)
        func(table_dir, fig_dir)
    print('  drawing plot_station', flush=True)
    plot_station(table_dir, fig_dir, allow_small)
    print('Figures: {0}'.format(fig_dir), flush=True)


def parse_args() -> argparse.Namespace:
    """
    Parse the command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Draw the figures of the 2411-station experiment'
    )
    parser.add_argument(
        '--allow-small', action='store_true',
        help='draw the station violins even with fewer than {0} '
        'stations'.format(MIN_STATIONS),
    )
    return parser.parse_args()


if __name__ == '__main__':
    print('Program plot_2411.py started', flush=True)
    args = parse_args()
    main(allow_small=args.allow_small)

