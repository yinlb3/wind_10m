#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the 10-day, 3-hourly wind product from per-station models.

Every station is handled independently. The model is trained on the full
52-slot window (3-240 h) read straight from ``data{sta}.npy`` plus the
hourly observations, the factors are expanded onto the uniform 80-slot
3-hourly grid. Each station npz stores float32 directions and speeds.

The main experiment trains on the 13-36 h window because that is what the
paper evaluates; this product is a separate deliverable and must not be
mixed with the tables of the main run.

Founded in 2026-11-04
Modified in 2026-11-20
@author: yinlb
"""

import argparse
import pathlib
import time
import typing

import numpy as np

from src import data_2411, window
from train_2411 import fit_lgb, predict_multi, sc_to_ds, uv_to_ds

MODELS = ('LightGBM-A', 'LightGBM-B')
OUTPUT_SUFFIX = {'LightGBM-A': '', 'LightGBM-B': '_b'}
PROGRESS_EVERY = 20


def train_station(
    x_val: np.ndarray,
    y_sc: np.ndarray,
    uv_val: np.ndarray,
    cfg: dict,
    models: typing.Sequence[str],
) -> typing.Dict[str, list]:
    """
    Fit the selected models on one station and the full 52-slot window.

    Args:
        x_val: Factors with shape (n_samples, 52, n_features).
        y_sc: Angular-encoding targets, shape (n_samples, 52, 3).
        uv_val: Vector targets, shape (n_samples, 52, 2).
        cfg: Configuration dictionary.
        models: Model names to fit.

    Returns:
        typing.Dict[str, list]: Fitted model lists keyed by model name.
    """
    out = {}
    flat_x = x_val.reshape(-1, x_val.shape[-1])
    if 'LightGBM-A' in models:
        flat_y = y_sc.reshape(-1, y_sc.shape[-1])
        keep = (
            (~np.isnan(flat_x).any(axis=1))
            & (~np.isnan(flat_y).any(axis=1))
        )
        if keep.sum() >= 100:
            out['LightGBM-A'] = fit_lgb(flat_x[keep], flat_y[keep], cfg)
    if 'LightGBM-B' in models:
        flat_uv = uv_val.reshape(-1, uv_val.shape[-1])
        keep = (
            (~np.isnan(flat_x).any(axis=1))
            & (~np.isnan(flat_uv).any(axis=1))
        )
        if keep.sum() >= 100:
            out['LightGBM-B'] = fit_lgb(flat_x[keep], flat_uv[keep], cfg)
    return out


def predict_station(
    full: np.ndarray,
    fitted: typing.Dict[str, list],
    marker: float,
) -> typing.Dict[str, np.ndarray]:
    """
    Predict every model on the expanded 80-slot grid.

    Args:
        full: Expanded factors, shape (n_samples, 80, n_features).
        fitted: Fitted model lists keyed by model name.
        marker: Calm-wind direction marker.

    Returns:
        typing.Dict[str, np.ndarray]: (direction, speed) arrays with shape
        (n_samples, 80, 2), float32.
    """
    valid = ~np.isnan(full).any(axis=2)
    shape = (full.shape[0], full.shape[1], 2)
    out = {}
    for name, models in fitted.items():
        block = np.full(shape, np.nan, dtype=np.float32)
        if valid.any():
            pred = predict_multi(models, full[valid])
            if name == 'LightGBM-A':
                block[valid] = sc_to_ds(pred, marker)
            else:
                block[valid] = uv_to_ds(pred, marker)
        out[name] = block
    return out


def main(
    limit: int = 0,
    models: typing.Sequence[str] = ('LightGBM-A',),
    out_dir: typing.Optional[str] = None,
) -> None:
    """
    Train, predict and save the 10-day product for every station.

    Args:
        limit: Number of stations to process (0 means all).
        models: Model names to build.
        out_dir: Output directory (default: artifacts/2411/deploy).
    """
    unknown = [name for name in models if name not in MODELS]
    if unknown or not models:
        print(
            'Unknown model: {0}; valid: {1}'.format(
                ', '.join(unknown) or '(empty)', ', '.join(MODELS)
            ),
            flush=True,
        )
        return
    cfg = data_2411.load_config()
    station_ids = data_2411.station_ids(cfg)
    if limit:
        station_ids = station_ids[:limit]
    total = len(station_ids)
    if out_dir:
        target = pathlib.Path(out_dir)
    else:
        target = pathlib.Path(cfg['paths']['pred_dir']).parent / 'deploy'
    target.mkdir(parents=True, exist_ok=True)
    marker = float(cfg['obs']['calm_marker'])
    weights = window.interp_matrix(cfg)
    required = {
        prefix + OUTPUT_SUFFIX[name]
        for name in models for prefix in ('dir', 'spd')
    }
    print(
        'Stations: {0}  models: {1}  out: {2}'.format(
            total, ', '.join(models), target
        ),
        flush=True,
    )

    start_all = time.time()
    built = 0
    for index, sta in enumerate(station_ids, 1):
        path = target / '{0}.npz'.format(sta)
        if path.exists():
            with np.load(path) as existing:
                if all(key in existing.files for key in required):
                    continue
        x_val = window.read_raw_factors(sta, cfg)
        direction, speed = window.full_observations(sta, cfg)
        if direction is None:
            print('  {0}: no observation, skipped'.format(sta), flush=True)
            continue
        fitted = train_station(
            x_val,
            data_2411.encode_sc(direction, speed, cfg),
            data_2411.encode_uv(direction, speed, cfg),
            cfg,
            models,
        )
        if not fitted:
            continue
        full = window.expand_leads(x_val, weights)
        arrays = {}
        for name, block in predict_station(full, fitted, marker).items():
            suffix = OUTPUT_SUFFIX[name]
            arrays['dir' + suffix] = block[..., 0]
            arrays['spd' + suffix] = block[..., 1].astype(np.float32)
        np.savez(path, **arrays)
        built += 1
        if index % PROGRESS_EVERY == 0 or index == total:
            elapsed = (time.time() - start_all) / 60
            rate = index / elapsed if elapsed > 0 else 0.0
            remain = (total - index) / rate if rate > 0 else 0.0
            print(
                '  [{0}/{1}] {2}  built={3}  {4:.1f} min  eta {5:.1f} min'
                .format(index, total, sta, built, elapsed, remain),
                flush=True,
            )
    print(
        'Finished: built={0} in {1:.1f} min'.format(
            built, (time.time() - start_all) / 60
        ),
        flush=True,
    )


if __name__ == '__main__':
    print('Program predict_10day.py started', flush=True)
    parser = argparse.ArgumentParser(
        description='Build the 10-day 3-hourly wind product'
    )
    parser.add_argument(
        '--limit', type=int, default=0,
        help='number of leading stations (0 means all, the default)',
    )
    parser.add_argument(
        '--models', type=str, default='LightGBM-A',
        help='LightGBM-A, LightGBM-B or both, comma separated',
    )
    parser.add_argument(
        '--out-dir', type=str, default='',
        help='output directory (default: artifacts/2411/deploy)',
    )
    args = parser.parse_args()
    names = [item.strip() for item in args.models.split(',') if item.strip()]
    main(limit=args.limit, models=names, out_dir=args.out_dir or None)

