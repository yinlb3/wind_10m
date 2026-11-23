#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Station quality control for the 2411-station experiment.

Read aligned observations from an existing cache when available, otherwise
read and clean the raw observation CSV directly. Write a per-station quality
table and keep stations whose missing fraction is within the threshold.

Founded in 2026-11-04
Modified in 2026-11-23
@author: yinlb
"""

import argparse
import pathlib
import typing

import numpy as np
import pandas as pd

from src import data_2411

# Observation missing fractions reported as a histogram.
BINS = (0.0, 0.1, 0.2, 0.3, 0.5, 1.01)
PROGRESS_EVERY = 200


def station_stats(sta: str, cfg: dict) -> typing.Optional[dict]:
    """
    Compute the observation quality statistics of one station.

    Args:
        sta: Station ID.
        cfg: Configuration dictionary.

    Returns:
        typing.Optional[dict]: Statistics, or None when observations are
        unavailable.
    """
    path = data_2411.cache_path(sta, cfg)
    if path.exists():
        with np.load(path) as data:
            direction = data['dir']
            speed = data['spd']
    else:
        direction, speed, _ = data_2411.load_obs(sta, cfg)
        if direction is None:
            return None
    nan_frac = float(np.isnan(speed).mean())
    calm_frac = float(
        (direction == cfg['obs']['calm_marker']).mean()
    )
    return {
        'station': sta,
        'n_sample': int(speed.shape[0]),
        'n_valid': int(np.sum(~np.isnan(speed))),
        'obs_nan_frac': round(nan_frac, 5),
        'calm_frac': round(calm_frac, 5),
    }


def main(max_missing: float = 0.30) -> None:
    """
    Write the station quality table and the kept-station list.

    Args:
        max_missing: Largest tolerated observation missing fraction.
    """
    cfg = data_2411.load_config()
    station_all = data_2411.available_station_ids(cfg)
    total = len(station_all)
    print('Stations to inspect: {0}'.format(total), flush=True)

    rows = list()
    absent = list()
    for index, sta in enumerate(station_all, 1):
        item = station_stats(sta, cfg)
        if item is None:
            absent.append(sta)
        else:
            rows.append(item)
        if index % PROGRESS_EVERY == 0 or index == total:
            print('  {0}/{1}'.format(index, total), flush=True)

    table = pd.DataFrame(rows).sort_values('station')
    qc_dir = pathlib.Path(cfg['paths']['result_dir']) / 'qc'
    qc_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(qc_dir / 'station_qc.csv', index=False, encoding='utf-8-sig')

    keep = table[table['obs_nan_frac'] <= max_missing]
    drop = table[table['obs_nan_frac'] > max_missing]
    list_path = cfg['paths'].get('station_list')
    if list_path:
        path = pathlib.Path(list_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as file_handle:
            file_handle.write('\n'.join(keep['station'].tolist()) + '\n')
        drop_path = qc_dir / 'stations_dropped.txt'
        with open(drop_path, 'w', encoding='utf-8') as file_handle:
            file_handle.write('\n'.join(drop['station'].tolist()) + '\n')
        print('Kept list: {0}'.format(path), flush=True)
        print('Dropped list: {0}'.format(drop_path), flush=True)

    counts, _ = np.histogram(table['obs_nan_frac'].values, bins=BINS)
    print('Missing-fraction histogram:', flush=True)
    for index, count in enumerate(counts, 1):
        print(
            '  [{0:.2f}, {1:.2f}): {2}'.format(
                BINS[index - 1], BINS[index], int(count)
            ),
            flush=True,
        )
    print(
        'Kept {0}  dropped {1} (>{2:.0%})  obs unavailable {3}'.format(
            keep.shape[0], drop.shape[0], max_missing, len(absent)
        ),
        flush=True,
    )
    print(
        'Median missing {0:.4f}  max kept {1:.4f}'.format(
            float(table['obs_nan_frac'].median()),
            float(keep['obs_nan_frac'].max()) if keep.shape[0]
            else float('nan'),
        ),
        flush=True,
    )
    print('QC table: {0}'.format(qc_dir / 'station_qc.csv'), flush=True)


if __name__ == '__main__':
    print('Program qc_stations_2411.py started', flush=True)
    parser = argparse.ArgumentParser(
        description='Run station quality control for the 2411 experiment'
    )
    parser.add_argument('--max-missing', type=float, default=0.30)
    args = parser.parse_args()
    main(max_missing=args.max_missing)
