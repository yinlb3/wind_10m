# -*- coding: utf-8 -*-
"""
Plotting helpers for evaluation/access97.py.

Founded in 2026-07-22
Modified in 2026-09-18
@author: yinlb
"""

import pathlib
import typing

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_rmse_improvement_bars(
    rmse_improve: np.ndarray,
    figure_dir: pathlib.Path,
    filename: str,
    x_labels: typing.List[str],
    x_label: str,
) -> None:
    """
    Draw grouped RMSE improvement bar charts.

    Args:
        rmse_improve (np.ndarray): Array of shape (3, n_groups).
        figure_dir (pathlib.Path): Directory to save figures.
        filename (str): Output filename.
        x_labels (typing.List[str]): Labels for the x-axis.
        x_label (str): X-axis title.
    """
    n_groups = rmse_improve.shape[1]
    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(n_groups) - 0.2,
        height=rmse_improve[0, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(n_groups),
        height=rmse_improve[1, :],
        width=0.2,
        color='yellow',
        label='MLR'
    )
    plt.bar(
        x=np.arange(n_groups) + 0.2,
        height=rmse_improve[2, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, n_groups))
    plt.xticks(range(n_groups), x_labels)
    plt.ylim((0, 45))
    plt.yticks((0, 5, 10, 15, 20, 25, 30, 35, 40, 45))
    plt.xlabel(x_label)
    plt.ylabel('RMSE改善率/%')
    plt.legend(loc='upper center', ncols=4)
    plt.savefig(figure_dir / filename, bbox_inches='tight', dpi=800)
    plt.close()
    print(np.min(rmse_improve), np.max(rmse_improve))


def plot_rmse_scatter(
    station_info: pd.DataFrame,
    rmse_improve_sta: np.ndarray,
    figure_dir: pathlib.Path,
    var_name: str,
    x_label: str,
    x_lims: typing.Tuple[float, float],
    x_ticks: typing.List[str],
    x_tick_labels: typing.List[str],
    filename: str,
) -> None:
    """
    Plot RMSE improvement versus a station geographic variable.

    Args:
        station_info (pd.DataFrame): Station metadata DataFrame.
        rmse_improve_sta (np.ndarray): Station-level RMSE improvements.
        figure_dir (pathlib.Path): Directory to save figures.
        var_name (str): Column name in station_info to use as x-axis.
        x_label (str): X-axis title.
        x_lims (typing.Tuple[float, float]): X-axis limits.
        x_ticks (typing.List[str]): X-axis tick positions.
        x_tick_labels (typing.List[str]): X-axis tick labels.
        filename (str): Output filename.
    """
    plt.figure(figsize=(5, 5))
    plt.scatter(
        x=station_info.loc[:, var_name],
        y=rmse_improve_sta[0, :],
        s=0.2,
        c='green',
        label='PDFM'
    )
    plt.scatter(
        x=station_info.loc[:, var_name],
        y=rmse_improve_sta[1, :],
        s=0.2,
        c='yellow',
        label='MLR'
    )
    plt.scatter(
        x=station_info.loc[:, var_name],
        y=rmse_improve_sta[2, :],
        s=0.2,
        c='red',
        label='LightGBM'
    )
    plt.xlim(x_lims)
    plt.xticks(x_ticks, x_tick_labels)
    plt.ylim((-20, 80))
    plt.yticks((-20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80))
    plt.xlabel(x_label)
    plt.ylabel('RMSE改善率/m')
    plt.legend()
    plt.savefig(figure_dir / filename, bbox_inches='tight', dpi=800)
    plt.close()
    print(x_label)
    for model_idx in range(3):
        print(
            np.corrcoef(
                station_info.loc[:, var_name],
                rmse_improve_sta[model_idx, :]
            )[0, 1]
        )


def plot_rmse_violin(
    rmse_improve_sta: np.ndarray,
    figure_dir: pathlib.Path,
) -> None:
    """
    Draw violin plot of station RMSE improvement distributions.

    Args:
        rmse_improve_sta (np.ndarray): Station-level RMSE improvements.
        figure_dir (pathlib.Path): Directory to save figures.
    """
    plt.figure(figsize=(5, 5))
    sns.violinplot(
        data={
            'PDFM': rmse_improve_sta[0, :],
            'MLR': rmse_improve_sta[1, :],
            'LightGBM': rmse_improve_sta[2, :]
        },
        palette=['green', 'yellow', 'red']
    )
    plt.xlabel('模型')
    plt.ylabel('RMSE改善率/m')
    plt.savefig(
        figure_dir / 'rmse_improvement_boxplot.jpg',
        bbox_inches='tight',
        dpi=800
    )
    plt.close()
