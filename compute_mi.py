#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the mutual-information matrix of the station factors.

Samples a subset of stations and forecast cycles, flattens the factors and
estimates the 26 x 26 mutual-information matrix with a 2-D histogram.

Founded in 2026-10-29
Modified in 2026-11-19
@author: yinlb
"""

import argparse
import pathlib
import typing

import numpy as np

from src import workflow

# Histogram bins used for the mutual-information estimate.
BINS = 32


def mutual_info(x_val: np.ndarray, y_val: np.ndarray) -> float:
    """
    Estimate the mutual information of two one-dimensional samples.

    Args:
        x_val: First sample.
        y_val: Second sample.

    Returns:
        float: Mutual information in nats.
    """
    hist, _, _ = np.histogram2d(x_val, y_val, bins=BINS)
    prob = hist / hist.sum()
    prob_x = prob.sum(axis=1)
    prob_y = prob.sum(axis=0)
    joint = prob > 0
    outer = prob_x[:, None] * prob_y[None, :]
    return float(np.sum(prob[joint] * np.log(prob[joint] / outer[joint])))


def collect_samples(
    cfg: dict, n_station: int, n_sample: int
) -> typing.Tuple[np.ndarray, typing.List[str]]:
    """
    Collect factor samples from a subset of stations.

    Args:
        cfg: Configuration dictionary.
        n_station: Number of stations to sample.
        n_sample: Number of forecast cycles to sample per station.

    Returns:
        typing.Tuple[np.ndarray, typing.List[str]]: Flattened factor matrix
        with shape (n, n_factors) and the factor names.
    """
    station_ids = workflow.station_ids(cfg)
    if n_station:
        station_ids = station_ids[::max(len(station_ids) // n_station, 1)]
    _, names = workflow.factor_columns(cfg)
    chunks = list()
    for sta in station_ids:
        arrays = workflow.load_arrays(sta, cfg)
        if arrays is None:
            continue
        factors = arrays['x']
        if n_sample:
            factors = factors[::max(factors.shape[0] // n_sample, 1)]
        chunks.append(factors.reshape(-1, factors.shape[-1]))
    return np.vstack(chunks), names


def main(n_station: int = 200, n_sample: int = 100) -> None:
    """
    Compute and save the mutual-information matrix.

    Args:
        n_station: Number of stations to sample (0 means all).
        n_sample: Number of forecast cycles to sample per station.
    """
    cfg = workflow.load_config()
    data, names = collect_samples(cfg, n_station, n_sample)
    valid = ~np.isnan(data).any(axis=1)
    data = data[valid].astype(np.float64)
    print('Samples: {0}  factors: {1}'.format(data.shape[0], len(names)),
          flush=True)

    n_factor = len(names)
    matrix = np.zeros((n_factor, n_factor), dtype=np.float64)
    for i in range(n_factor):
        for j in range(i, n_factor):
            value = mutual_info(data[:, i], data[:, j])
            matrix[i, j] = value
            matrix[j, i] = value
        if (i + 1) % 5 == 0:
            print('  {0}/{1}'.format(i + 1, n_factor), flush=True)

    result_dir = pathlib.Path(cfg['paths']['result_dir'])
    table_dir = result_dir / 'tables'
    table_dir.mkdir(parents=True, exist_ok=True)
    np.save(table_dir / 'mi.npy', matrix)
    with open(table_dir / 'mi_names.txt', 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(names))
    print('Saved {0}'.format(table_dir / 'mi.npy'), flush=True)


if __name__ == '__main__':
    print('Program compute_mi.py started', flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--stations', type=int, default=200)
    parser.add_argument('--samples', type=int, default=100)
    args = parser.parse_args()
    main(n_station=args.stations, n_sample=args.samples)
