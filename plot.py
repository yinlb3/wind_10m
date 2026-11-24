#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw the figures of the station experiment.

Reads the CSV tables written by verify.py plus the mutual-information
matrix written by compute_mi.py, and saves the figures to the result
directory.

Founded in 2026-10-29
Modified in 2026-11-24
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

from src import workflow  # noqa: E402

# The manuscript compares one LightGBM model with the three baselines; the
# vector branch of the angle experiment appears in Table 4 only.
MODELS = ['ECMWF-IFS', 'PDFM', 'MLR', 'LightGBM']
COLORS = ['blue', 'green', 'yellow', 'red']
RETIRED = 'LightGBM-B'
SECTORS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
SPEED_LABELS = [
    '≥1级', '≥2级', '≥3级', '≥4级', '≥5级', '≥6级', '≥7级', '≥8级',
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

plt.rcParams['font.family'] = 'sans-serif'
# SimHei first so that every Chinese glyph has a real font; the Latin labels
# and the units in math mode keep the Times look through the STIX fontset.
plt.rcParams['font.sans-serif'] = [
    'SimHei', 'Microsoft YaHei', 'DejaVu Sans'
]
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
_CFG = workflow.load_config()
FIG_DPI = int(_CFG['figure']['dpi'])
FIG_FORMATS = [
    str(item).lstrip('.').lower() for item in _CFG['figure']['formats']
]
if not FIG_FORMATS:
    raise ValueError('figure.formats is empty in the active configuration')
# The canvas matches the printed size: one sub figure is 7.5 cm wide (half of
# a 15 cm two-column figure) and a single figure is 7.8 cm. Keeping the canvas
# at the printed size is what makes one type size readable in every figure.
CM = 2.54
WIDE_SIZE = (7.5 / CM, 4.6 / CM)
SQUARE_SIZE = (7.5 / CM, 7.5 / CM)
RADAR_SIZE = (7.5 / CM, 7.5 / CM)
DEFAULT_SIZE = (6.4, 4.8)
# One type scale for every figure.
FS_LABEL = 7
FS_TICK = 6.5
FS_LEGEND = 6.5

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
    kwargs.setdefault('bbox_inches', 'tight')
    kwargs.setdefault('pad_inches', 0.02)
    for ext in FIG_FORMATS:
        path = fig_dir / '{0}.{1}'.format(name, ext)
        fig.savefig(path, dpi=FIG_DPI, **kwargs)


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
    table = pd.read_csv(path, encoding='utf-8-sig')
    table = table.rename(columns={'LightGBM-A': 'LightGBM'})
    table = table.drop(columns=[RETIRED], errors='ignore')
    if 'model' in table.columns:
        table = table[table['model'] != RETIRED]
        table['model'] = table['model'].replace({'LightGBM-A': 'LightGBM'})
    return table


def plot_mi(result_dir: pathlib.Path, fig_dir: pathlib.Path) -> None:
    """
    Draw the mutual-information heat map (Figure 1).

    The factors are re-sorted into the level-by-level order used by the
    paper before the matrix is drawn.

    Args:
        result_dir: Directory holding mi.npy.
        fig_dir: Output directory for figures.
    """
    path = result_dir / 'mi.npy'
    names_path = result_dir / 'mi_names.txt'
    if not path.exists() or not names_path.exists():
        print('  missing mi.npy', flush=True)
        return
    matrix = np.load(path)
    names = names_path.read_text(encoding='utf-8').split()
    index = [names.index(name) for name in FACTOR_ORDER if name in names]
    matrix = matrix[np.ix_(index, index)]
    labels = [FACTOR_LABELS[names[item]] for item in index]

    fig, axis = plt.subplots(figsize=DEFAULT_SIZE)
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
    fig.colorbar(image, ax=axis, shrink=0.75).ax.tick_params(
        labelsize=FS_TICK
    )
    fig.tight_layout(pad=0.1)
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
    axis.set_xlabel(xlabel, fontsize=FS_LABEL)
    axis.set_ylabel(ylabel, fontsize=FS_LABEL)
    axis.tick_params(labelsize=FS_TICK)
    if ylim is not None:
        axis.set_ylim(0, ylim)
    if legend == 'inside':
        axis.legend(
            fontsize=FS_LEGEND, loc='upper right', frameon=False,
            handlelength=1.2, handletextpad=0.4, borderpad=0.2,
        )
    else:
        axis.legend(
            loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(names),
            fontsize=FS_LEGEND, frameon=False, borderaxespad=0.1,
            handlelength=1.2, handleheight=0.8, handletextpad=0.4,
            columnspacing=1.0,
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

    fig, axis = plt.subplots(figsize=WIDE_SIZE)
    _grouped_bars(
        axis, values, SPEED_LABELS, 'TS', '风速等级', ylim=1.0,
        legend='top',
    )
    fig.tight_layout(pad=0.1)
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
    fig, axis = plt.subplots(figsize=WIDE_SIZE)
    _grouped_bars(
        axis, table, ticks, 'kappa', '风速等级', ylim=1.0, legend='top',
    )
    fig.tight_layout(pad=0.1)
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
        axis, table, ticks, ylabel, xlabel, ylim=ylim, legend='top'
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

    items = (
        (year_ws, year_ticks, '年份', '风速 RMSE（m·s$^{-1}$）', None,
         'fig3a_year_speed'),
        (year_wd, year_ticks, '年份', '风向 RMSE（°）', 100.0,
         'fig3b_year_dir'),
        (month, month_ticks, '月份', '风速 RMSE（m·s$^{-1}$）', None,
         'fig3c_month_speed'),
        (month_dir, month_ticks, '月份', '风向 RMSE（°）', 100.0,
         'fig3d_month_dir'),
    )
    for table, ticks, xlabel, ylabel, ylim, name in items:
        fig, axis = plt.subplots(figsize=WIDE_SIZE)
        _panel(axis, table, ticks, xlabel, ylabel, ylim)
        fig.tight_layout(pad=0.1)
        save_fig(fig, fig_dir, name)
        plt.close(fig)


def _bar_panel(
    axis: plt.Axes,
    table: pd.DataFrame,
    x_val: np.ndarray,
    xlabel: str,
    ylabel: str,
    ticks: typing.Optional[typing.Sequence[float]] = None,
    labels: typing.Optional[typing.Sequence[str]] = None,
    group_width: float = 0.8,
    span: typing.Optional[typing.Tuple[float, float]] = None,
) -> None:
    """
    Draw one grouped bar panel, one bar per model and group.

    Args:
        axis: Target axis.
        table: Table with one column per model and one row per group.
        x_val: X coordinates of the rows.
        xlabel: X axis label.
        ylabel: Y axis label.
        ticks: Tick positions, one per day or per forecast hour.
        labels: Tick labels matching ticks.
        group_width: Width of one group of bars in x units.
        span: X axis limits, computed from the data when None.
    """
    names = [model for model in MODELS if model in table.columns]
    width = group_width / max(len(names), 1)
    for index, model in enumerate(names):
        offset = (index - (len(names) - 1) / 2.0) * width
        axis.bar(
            x_val + offset, table[model].to_numpy(dtype=float),
            width=width, color=COLORS[MODELS.index(model)], label=model,
        )
    if ticks is not None:
        axis.set_xticks(ticks, labels)
    if span is not None:
        axis.set_xlim(span)
    axis.set_xlabel(xlabel, fontsize=FS_LABEL)
    axis.set_ylabel(ylabel, fontsize=FS_LABEL)
    axis.tick_params(labelsize=FS_TICK)
    axis.grid(axis='y', alpha=0.3, linewidth=0.6)
    axis.legend(
        fontsize=FS_LEGEND, ncol=len(names), loc='lower center',
        bbox_to_anchor=(0.5, 1.0), frameon=False, borderaxespad=0.1,
        handlelength=1.2, handleheight=0.8, handletextpad=0.4,
        columnspacing=1.0,
    )


def plot_rmse_lines(result_dir: pathlib.Path, fig_dir: pathlib.Path) -> None:
    """
    Draw the lead-time and hour-of-day RMSE as bar charts (fig3e-3h).

    The lead time is grouped by day, with one bar per 3 h slot, so the upward
    drift with lead time and the diurnal cycle are both visible.

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
    # Average the eight 3 h slots of every day, so the bars show the drift
    # with lead time in day units instead of 80 crowded slots.
    per_day = 8
    days = int(np.ceil(lead.shape[0] / float(per_day)))
    lead = lead.groupby(np.arange(lead.shape[0]) // per_day).mean()
    lead_dir = lead_dir.groupby(
        np.arange(lead_dir.shape[0]) // per_day
    ).mean()
    lead_x = np.arange(lead.shape[0], dtype=float)
    day_ticks = lead_x
    day_labels = ['{0}'.format(item + 1) for item in range(days)]
    hour_x = hour.index.to_numpy(dtype=float)
    keep = hour.notna().any(axis=1).to_numpy()
    hour = hour[keep]
    hour_dir = hour_dir[keep]
    hour_x = hour_x[keep]
    hour_ticks = hour_x
    hour_labels = ['{0}'.format(int(item)) for item in hour_ticks]
    items = (
        (lead, 'fig3e_lead_speed', '预报时效（d）',
         '风速 RMSE（m·s$^{-1}$）', day_ticks, day_labels, 0.7, None),
        (lead_dir, 'fig3f_lead_dir', '预报时效（d）', '风向 RMSE（°）',
         day_ticks, day_labels, 0.7, None),
        (hour, 'fig3g_hour_speed', '预报时刻（UTC）',
         '风速 RMSE（m·s$^{-1}$）', hour_ticks, hour_labels,
         2.0, (hour_x.min() - 1.4, hour_x.max() + 1.4)),
        (hour_dir, 'fig3h_hour_dir', '预报时刻（UTC）', '风向 RMSE（°）',
         hour_ticks, hour_labels, 2.0,
         (hour_x.min() - 1.4, hour_x.max() + 1.4)),
    )
    for table, name, xlabel, ylabel, ticks, labels, span, limits in items:
        fig, axis = plt.subplots(figsize=WIDE_SIZE)
        if name.startswith('fig3e') or name.startswith('fig3f'):
            x_val = lead_x
        else:
            x_val = hour_x
        _bar_panel(
            axis, table, x_val, xlabel, ylabel, ticks, labels,
            group_width=span, span=limits,
        )
        fig.tight_layout(pad=0.1)
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
        axis.plot([pos, pos], [low, high], color='black', linewidth=1.2)
        axis.add_patch(
            plt.Rectangle(
                (pos - 0.04, quar1), 0.08, quar3 - quar1,
                facecolor='black', edgecolor='black',
            )
        )
        axis.plot(
            [pos - 0.04, pos + 0.04], [median, median],
            color='white', linewidth=1.2,
        )
    axis.set_xticks(positions, names)
    axis.set_xlim(0.4, len(names) + 0.6)
    axis.set_xlabel('模型', fontsize=FS_LABEL)
    axis.set_ylabel(label, fontsize=FS_LABEL)
    axis.tick_params(labelsize=FS_TICK)


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
            for stem in (
                'fig4a_station_speed', 'fig4b_station_dir',
                'fig4c_station_speed_improve', 'fig4d_station_dir_improve',
            ):
                stale = fig_dir / '{0}.{1}'.format(stem, ext)
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
    improved = [model for model in MODELS[1:] if model in ws.columns]
    base_ws = ws['ECMWF-IFS'].to_numpy(dtype=float)
    base_wd = wd['ECMWF-IFS'].to_numpy(dtype=float)
    items = (
        ([ws[name].dropna().to_numpy() for name in names], names,
         '风速 RMSE（m·s$^{-1}$）', 'fig4a_station_speed'),
        ([wd[name].dropna().to_numpy() for name in names], names,
         '风向 RMSE（°）', 'fig4b_station_dir'),
        ([(base_ws - ws[name].to_numpy(dtype=float)) / base_ws * 100.0
          for name in improved], improved,
         '风速 RMSE 改善率（%）', 'fig4c_station_speed_improve'),
        ([(base_wd - wd[name].to_numpy(dtype=float)) / base_wd * 100.0
          for name in improved], improved,
         '风向 RMSE 改善率（%）', 'fig4d_station_dir_improve'),
    )
    for data, labels, ylabel, name in items:
        data = [item[~np.isnan(item)] for item in data]
        fig, axis = plt.subplots(figsize=SQUARE_SIZE)
        _violin(axis, data, labels, ylabel)
        fig.tight_layout(pad=0.1)
        save_fig(fig, fig_dir, name)
        plt.close(fig)


def _radar(
    axis: plt.Axes, table: pd.DataFrame,
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
    axis.tick_params(axis='x', labelsize=FS_LABEL)
    for label in axis.get_yticklabels():
        label.set_fontsize(FS_TICK)
        label.set_bbox({'facecolor': 'white', 'edgecolor': 'none',
                        'pad': 1.0})
    axis.set_rlabel_position(22.5)
    axis.grid(alpha=0.6)


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

    items = (
        (speed, base_speed, 'fig5a_radar_speed'),
        (direction, base_dir, 'fig5b_radar_dir'),
        (speed, base_speed, 'fig5c_radar_speed_improve'),
        (direction, base_dir, 'fig5d_radar_dir_improve'),
    )
    for table, base, name in items:
        fig = plt.figure(figsize=RADAR_SIZE)
        axis = fig.add_subplot(1, 1, 1, projection='polar')
        if 'improve' not in name:
            base = None
        _radar(axis, table, base)
        fig.tight_layout(pad=0.1)
        save_fig(fig, fig_dir, name)
        plt.close(fig)


def main(allow_small: bool = False) -> None:
    """
    Draw all available figures.

    Args:
        allow_small: Draw the station violins even when fewer than
            MIN_STATIONS stations are available.
    """
    cfg = _CFG
    result_dir = workflow.result_root(cfg)
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
        description='Draw the figures of the station experiment'
    )
    parser.add_argument(
        '--allow-small', action='store_true',
        help='draw the station violins even with fewer than {0} '
        'stations'.format(MIN_STATIONS),
    )
    return parser.parse_args()


if __name__ == '__main__':
    print('Program plot.py started', flush=True)
    args = parse_args()
    main(allow_small=args.allow_small)

