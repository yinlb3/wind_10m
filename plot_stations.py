#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw the spatial distribution of the retained and dropped stations.

The station list comes from meteva's bundled station table, which holds
only positions, so every point is drawn without any value.

Founded in 2026-11-21
Modified in 2026-11-21
@author: yinlb
"""

import pathlib

import matplotlib
import meteva.base as meb

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from src import workflow  # noqa: E402

CM = 1.0 / 2.54


def _read_ids(path: pathlib.Path) -> set:
    """
    Read a station-id list.

    Args:
        path: Text file with one station id per line.

    Returns:
        set: Station ids as strings.
    """
    with open(path, 'r', encoding='utf-8') as file_handle:
        return {line.strip() for line in file_handle if line.strip()}


def main() -> None:
    """Draw and save the station map."""
    cfg = workflow.load_config()
    dpi = int(cfg['figure']['dpi'])
    formats = [
        str(item).lstrip('.').lower() for item in cfg['figure']['formats']
    ]
    result_dir = pathlib.Path(cfg['paths']['result_dir'])
    fig_dir = workflow.result_root(cfg) / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams['font.sans-serif'] = [
        'SimHei', 'Microsoft YaHei', 'DejaVu Sans'
    ]
    plt.rcParams['axes.unicode_minus'] = False

    table = meb.read_station(meb.station_国家站)
    table['id'] = table['id'].astype(str)
    qc_dir = result_dir / 'qc'
    keep = _read_ids(qc_dir / 'stations_ok.txt')
    drop = _read_ids(qc_dir / 'stations_dropped.txt')
    keep_rows = table[table['id'].isin(keep)]
    drop_rows = table[table['id'].isin(drop)]

    width = 7.8 * CM
    fig, axis = plt.subplots(figsize=(width, width * 0.85))
    axis.scatter(
        keep_rows['lon'], keep_rows['lat'], s=1.0, c='tab:blue',
        linewidths=0, label='保留站',
    )
    axis.scatter(
        drop_rows['lon'], drop_rows['lat'], s=6.0, c='tab:red',
        marker='x', linewidths=0.6, label='剔除站',
    )
    axis.set_xlabel('经度（°E）', fontsize=7)
    axis.set_ylabel('纬度（°N）', fontsize=7)
    axis.tick_params(labelsize=6.5)
    axis.legend(fontsize=5.5, markerscale=2.5, frameon=False)
    axis.set_aspect('equal', adjustable='datalim')
    fig.tight_layout()
    for ext in formats:
        path = fig_dir / 'fig6_stations.{0}'.format(ext)
        fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(
        'kept={0}  dropped={1}  figures: {2}'.format(
            keep_rows.shape[0], drop_rows.shape[0], fig_dir
        ),
        flush=True,
    )


if __name__ == '__main__':
    print('Program plot_stations.py started', flush=True)
    main()
