#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build optional per-station caches for the station experiment.

The configured cache mode selects observation-only or full feature storage.
Files are uncompressed NPZ archives; encoded targets are reconstructed on
loading instead of being duplicated on disk.

Founded in 2026-10-29
Modified in 2026-11-23
@author: yinlb
"""

import argparse
import json
import pathlib
import time
import typing

import numpy as np

from src import workflow

# Progress print interval in stations.
PROGRESS_EVERY = 20


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Build per-station caches for the configured experiment'
    )
    parser.add_argument(
        '--limit', type=int, default=0,
        help='number of randomly sampled stations (0 means all)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='random seed for the station sampling (default: 42)'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Rebuild caches that already exist'
    )
    return parser.parse_args()


def _cache_ready(path: pathlib.Path, mode: str, cfg: dict) -> bool:
    """Return whether an existing cache satisfies the active configuration."""
    if not path.exists():
        return False
    n_hour = int(workflow.target_hours(cfg).size)
    with np.load(path) as data:
        if 'dir' not in data.files or 'spd' not in data.files:
            return False
        if data['dir'].shape[1] != n_hour:
            return False
        if mode == 'full':
            if 'x' not in data.files or data['x'].shape[1] != n_hour:
                return False
        if 'meta' in data.files:
            return str(data['meta']) == workflow.cache_descriptor(cfg)
    return True


def _summarize(arrays: dict, cfg: dict) -> dict:
    """
    Compute quality statistics of one station.

    Args:
        arrays: Arrays returned by the selected cache builder.
        cfg: Configuration dictionary.

    Returns:
        dict: Quality statistics.
    """
    direction = arrays['dir']
    speed = arrays['spd']
    calm = direction == cfg['obs']['calm_marker']
    item = {
        'calm_frac': round(float(calm.mean()), 5),
        'obs_nan_frac': round(float(np.isnan(speed).mean()), 5),
        'speed_max': round(float(np.nanmax(speed)), 3),
    }
    factors = arrays.get('x')
    if factors is None:
        item['x_all_nan_frac'] = None
        item['samples_all_nan'] = None
    else:
        all_nan = np.isnan(factors).all(axis=2)
        item['x_all_nan_frac'] = round(float(all_nan.mean()), 5)
        item['samples_all_nan'] = int(all_nan.all(axis=1).sum())
    return item


def main(
    limit: int = 0, force: bool = False, seed: int = 42,
) -> typing.List[dict]:
    """
    Build caches for every station with both factors and observations.

    A limited run samples stations at random with a fixed seed, matching the
    sampling of benchmark_schemes.py so that a small run reuses these caches.

    Args:
        limit: Number of randomly sampled stations (0 means all).
        force: Whether to rebuild existing caches.
        seed: Random seed used for the station sampling.

    Returns:
        typing.List[dict]: Per-station quality report.
    """
    cfg = workflow.load_config()
    mode = workflow.cache_mode(cfg)
    if mode == 'none':
        print(
            "Cache building is disabled. Set cache.mode to 'obs' or 'full' "
            'before running this script.',
            flush=True,
        )
        return []

    station_ids = workflow.available_ids(cfg)
    if limit and limit < len(station_ids):
        rng = np.random.default_rng(seed)
        picked = sorted(rng.choice(len(station_ids), limit, replace=False))
        station_ids = [station_ids[index] for index in picked]
    total = len(station_ids)
    if total == 0:
        print('No station found', flush=True)
        return []

    print(
        f'Stations: {total}  mode: {mode}  '
        f'cache_dir: {cfg["paths"]["cache_dir"]}',
        flush=True,
    )
    report = list()
    skipped = 0
    start_all = time.time()

    for index, sta in enumerate(station_ids, 1):
        path = workflow.cache_path(sta, cfg)
        if not force and _cache_ready(path, mode, cfg):
            skipped += 1
            continue
        if mode == 'obs':
            arrays = workflow.build_observations(sta, cfg)
        else:
            arrays = workflow.build_station(sta, cfg)
        if arrays is None:
            print(f'  {sta}: no observation, skipped', flush=True)
            continue
        workflow.save_cache(sta, cfg, arrays)
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
    print('Program build_cache.py started', flush=True)
    args = parse_args()
    main(limit=args.limit, force=args.force, seed=args.seed)
