#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build per-station caches for the 2411-station experiment.

For every station with both NWP factors and observations, read the factor
array, clean and align the observations, and write one compressed npz file
containing the interpolated factors and the encoded targets.

Founded in 2026-10-29
Modified in 2026-11-19
@author: yinlb
"""

import argparse
import json
import pathlib
import time
import typing

import numpy as np

from src import data_2411

# Progress print interval in stations.
PROGRESS_EVERY = 20


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Build per-station caches for the 2411 experiment'
    )
    parser.add_argument(
        '--limit', type=int, default=0,
        help='Process only the first N stations (0 means all)'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Rebuild caches that already exist'
    )
    return parser.parse_args()


def _summarize(arrays: dict, cfg: dict) -> dict:
    """
    Compute quality statistics of one station cache.

    Args:
        arrays: Arrays returned by data_2411.build_station.
        cfg: Configuration dictionary.

    Returns:
        dict: Quality statistics.
    """
    factors = arrays['x']
    targets = arrays['y']
    speed = arrays['spd']
    calm = targets[..., 0] == cfg['obs']['calm_marker']
    all_nan = np.isnan(factors).all(axis=2)
    return {
        'calm_frac': round(float(calm.mean()), 5),
        'obs_nan_frac': round(float(np.isnan(speed).mean()), 5),
        'x_all_nan_frac': round(float(all_nan.mean()), 5),
        'samples_all_nan': int(all_nan.all(axis=1).sum()),
        'speed_max': round(float(np.nanmax(speed)), 3),
    }


def main(limit: int = 0, force: bool = False) -> typing.List[dict]:
    """
    Build caches for every station with both factors and observations.

    Args:
        limit: Number of stations to process (0 means all).
        force: Whether to rebuild existing caches.

    Returns:
        typing.List[dict]: Per-station quality report.
    """
    cfg = data_2411.load_config()
    station_ids = data_2411.available_station_ids(cfg)
    if limit:
        station_ids = station_ids[:limit]
    total = len(station_ids)
    if total == 0:
        print('No station found', flush=True)
        return []

    print(
        f'Stations: {total}  cache_dir: {cfg["paths"]["cache_dir"]}',
        flush=True,
    )
    report = list()
    skipped = 0
    start_all = time.time()

    for index, sta in enumerate(station_ids, 1):
        path = data_2411.cache_path(sta, cfg)
        if path.exists() and not force:
            skipped += 1
            continue
        arrays = data_2411.build_station(sta, cfg)
        if arrays is None:
            print(f'  {sta}: no observation, skipped', flush=True)
            continue
        data_2411.save_cache(sta, cfg, arrays)
        item = _summarize(arrays, cfg)
        item['sta'] = sta
        report.append(item)

        if index % PROGRESS_EVERY == 0 or index == total:
            elapsed = time.time() - start_all
            done = index - skipped
            rate = done / elapsed if elapsed > 0 else 0.0
            remain = (total - index) / rate / 60 if rate > 0 else 0.0
            print(
                f'[{index}/{total}] {sta}  done={done} skipped={skipped}  '
                f'{rate:.2f} sta/s  elapsed={elapsed / 60:.1f} min  '
                f'eta={remain:.1f} min',
                flush=True,
            )

    result_dir = pathlib.Path(cfg['paths']['result_dir']) / 'qc'
    result_dir.mkdir(parents=True, exist_ok=True)
    report_path = result_dir / 'cache_report.json'
    with open(report_path, 'w', encoding='utf-8') as file_handle:
        json.dump(report, file_handle, ensure_ascii=False, indent=1)

    total_minutes = (time.time() - start_all) / 60
    print(
        f'Finished: built={len(report)} skipped={skipped} '
        f'total={total_minutes:.1f} min',
        flush=True,
    )
    print(f'Report: {report_path}', flush=True)
    return report


if __name__ == '__main__':
    print('Program build_cache_2411.py started', flush=True)
    args = parse_args()
    main(limit=args.limit, force=args.force)
