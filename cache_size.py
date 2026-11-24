#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate optional cache sizes for the configured wind dataset.

This compatibility entry point uses the generic adapter and cache estimator.
It does not create or modify files.

Founded in 2026-11-23
Modified in 2026-11-23
@author: yinlb
"""

import argparse
import pathlib
import typing

from src import workflow
from wind10m.cache import estimate_sizes, format_size
from wind10m.config import load_config
from wind10m.dataset import create_adapter


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Estimate cache sizes without reading array payloads'
    )
    parser.add_argument(
        '--stations', type=int, default=0,
        help='Station count (0 reads the QC list or raw-data intersection)',
    )
    parser.add_argument(
        '--samples', type=int, default=0,
        help='Forecast samples per station (0 uses constants.n_samples)',
    )
    parser.add_argument(
        '--hours', type=int, default=0,
        help='Target hours per sample (0 uses the active lead mode)',
    )
    parser.add_argument(
        '--features', type=int, default=0,
        help='Feature count (0 uses the configured factor layout)',
    )
    return parser.parse_args()


def infer_station_count(cfg: dict) -> typing.Tuple[int, str]:
    """
    Infer a station count from the QC list or available raw data.

    Args:
        cfg: Configuration dictionary.

    Returns:
        typing.Tuple[int, str]: Station count and its source.

    Raises:
        ValueError: If no station count can be determined.
    """
    list_path = cfg['paths'].get('station_list')
    if list_path:
        path = pathlib.Path(list_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as file_handle:
                stations = {
                    line.strip() for line in file_handle if line.strip()
                }
            if stations:
                return len(stations), 'QC list: {0}'.format(path)
    stations = create_adapter(cfg).available_ids()
    if stations:
        return len(stations), 'Raw-data station intersection'
    raise ValueError('No stations found; provide --stations N')


def main() -> None:
    """Print cache size estimates for the selected dataset dimensions."""
    args = parse_args()
    values = (args.stations, args.samples, args.hours, args.features)
    if any(value < 0 for value in values):
        raise ValueError('Dimension overrides must be non-negative')

    cfg = load_config()
    adapter = create_adapter(cfg)
    if args.stations:
        stations = args.stations
        source = 'Command-line override'
    else:
        stations, source = infer_station_count(cfg)
    samples = args.samples or adapter.n_samples
    hours = args.hours or int(adapter.target_hours().size)
    features = args.features or adapter.feature_count()
    if min(stations, samples, hours, features) <= 0:
        raise ValueError('All estimated dimensions must be positive')

    sizes = estimate_sizes(stations, samples, hours, features)
    print('Cache size estimate (float32, uncompressed)')
    print('Station source: {0}'.format(source))
    print(
        'Dimensions: {0} stations x {1} samples x {2} hours; '
        '{3} features'.format(stations, samples, hours, features)
    )
    print('Mode  Per station         Total')
    for mode in workflow.CACHE_MODES:
        total = sizes[mode]
        per_station = total // stations
        print(
            '{0:<4}  {1:>15}  {2:>15}'.format(
                mode, format_size(per_station), format_size(total)
            )
        )
    print('Configured cache.mode: {0}'.format(workflow.cache_mode(cfg)))
    print('obs: stores dir/spd/pre and rebuilds x/y/uv on demand')
    print('full: stores x/dir/spd/pre and rebuilds y/uv on demand')
    print('Estimates exclude small NPZ metadata overhead.')


if __name__ == '__main__':
    print('Program cache_size.py started', flush=True)
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit('Cache estimate failed: {0}'.format(exc))
