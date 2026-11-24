#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the station correction models with the per-station scheme.

For every leave-one-year-out fold and every station, fit five models:
the raw ECMWF-IFS wind (no training), a quantile-mapping PDFM baseline,
a multiple linear regression baseline, the angular-encoding LightGBM and
the vector-decomposition LightGBM. Predictions are written as
(direction, speed) arrays with shape (n_samples, n_hours, n_stations, 2).

Founded in 2026-10-29
Modified in 2026-11-19
@author: yinlb
"""

import argparse
import csv
import os
import pathlib
import time
import typing
from concurrent.futures import ProcessPoolExecutor

import lightgbm as lgbm
import numpy as np
from sklearn import linear_model as lm

from src import workflow

MODELS = ('ECMWF-IFS', 'PDFM', 'MLR', 'LightGBM-A', 'LightGBM-B')
# Threads per worker; measured throughput peaks near seven on this machine.
WORKER_THREADS = 7
PDFM_QUANTILES = np.linspace(0.0, 1.0, 101)
CALM_SPEED = 0.3


def uv_to_ds(uv: np.ndarray, marker: float) -> np.ndarray:
    """
    Convert u/v components to (direction, speed).

    Args:
        uv: Array with u and v in the last dimension.
        marker: Calm-wind direction marker.

    Returns:
        np.ndarray: Array with direction and speed in the last dimension.
    """
    u_val = uv[..., 0]
    v_val = uv[..., 1]
    speed = np.sqrt(u_val ** 2 + v_val ** 2)
    calm = speed < CALM_SPEED
    direction = (180.0 + np.degrees(np.arctan2(u_val, v_val))) % 360.0
    direction = np.where(calm, marker, direction)
    return np.stack([direction, speed], axis=-1).astype(np.float32)


def sc_to_ds(sc: np.ndarray, marker: float) -> np.ndarray:
    """
    Convert (speed, sin, cos) predictions to (direction, speed).

    Args:
        sc: Array with speed, sine and cosine in the last dimension.
        marker: Calm-wind direction marker.

    Returns:
        np.ndarray: Array with direction and speed in the last dimension.
    """
    speed = sc[..., 0].astype(np.float32).copy()
    calm = (speed == marker) | (speed < CALM_SPEED)
    speed = np.where(calm, np.maximum(speed, CALM_SPEED), speed)
    direction = np.degrees(np.arctan2(sc[..., 1], sc[..., 2])) % 360.0
    direction = np.where(calm, marker, direction)
    return np.stack([direction, speed], axis=-1).astype(np.float32)


def quantile_map(
    obs: np.ndarray, fcst: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Fit a quantile-mapping table from forecasts to observations.

    Args:
        obs: Observed values, one-dimensional.
        fcst: Forecast values, one-dimensional.

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: Observation quantiles and the
        matching, strictly increasing forecast quantiles.
    """
    obs_q = np.quantile(obs, PDFM_QUANTILES)
    fcst_q = np.quantile(fcst, PDFM_QUANTILES)
    fcst_q, index = np.unique(fcst_q, return_index=True)
    return obs_q[index], fcst_q


def fit_lgb(
    tr_x: np.ndarray, tr_y: np.ndarray, cfg: dict
) -> typing.List[lgbm.LGBMRegressor]:
    """
    Fit one LightGBM regressor per target column.

    Args:
        tr_x: Training features, shape (n, n_features).
        tr_y: Training targets, shape (n, n_targets).
        cfg: Configuration dictionary.

    Returns:
        typing.List[lgbm.LGBMRegressor]: Fitted models.
    """
    params = dict(cfg['model'].get('lgb_params') or {})
    models = list()
    for k in range(tr_y.shape[-1]):
        model = lgbm.LGBMRegressor(
            n_jobs=cfg['model']['lgb_n_jobs'],
            verbose=-1,
            force_col_wise=True,
            random_state=cfg['model']['random_seed'],
            **params,
        )
        model.fit(tr_x, tr_y[:, k])
        models.append(model)
    return models


def predict_multi(
    models: typing.List[typing.Any], data: np.ndarray
) -> np.ndarray:
    """
    Predict every target column with its own model.

    Args:
        models: Fitted models, one per target.
        data: Features, shape (n, n_features).

    Returns:
        np.ndarray: Predictions, shape (n, n_targets).
    """
    columns = [model.predict(data) for model in models]
    return np.stack(columns, axis=-1)


def fit_mlr(
    tr_x: np.ndarray, tr_y: np.ndarray
) -> typing.List[lm.LinearRegression]:
    """
    Fit one multiple linear regression per target column.

    Args:
        tr_x: Training features.
        tr_y: Training targets.

    Returns:
        typing.List[lm.LinearRegression]: Fitted models.
    """
    models = list()
    for k in range(tr_y.shape[-1]):
        model = lm.LinearRegression()
        model.fit(tr_x, tr_y[:, k])
        models.append(model)
    return models


FILE_NAMES = {
    'ECMWF-IFS': 'ecmwf_ifs',
    'PDFM': 'pdfm',
    'MLR': 'mlr',
    'LightGBM-A': 'lgb_a',
    'LightGBM-B': 'lgb_b',
}


def process_station(
    arr: dict,
    cfg: dict,
    year_start: int,
    year_end: int,
    train_index: np.ndarray,
    selected: typing.Sequence[str],
    timing: typing.Optional[dict] = None,
) -> typing.Dict[str, np.ndarray]:
    """
    Fit and predict all selected models for one station and one fold.

    Args:
        arr: Cached arrays of the station.
        cfg: Configuration dictionary.
        year_start: First sample index of the held-out year.
        year_end: End sample index of the held-out year, exclusive.
        train_index: Sample indices of the training years.
        selected: Model names to run.
        timing: Optional accumulator for per-model wall-clock seconds.

    Returns:
        typing.Dict[str, np.ndarray]: Predictions as (direction, speed)
        arrays with shape (n_samples, n_hours, 2).
    """
    marker = float(cfg['obs']['calm_marker'])

    def _tick(name: str, mark: float) -> None:
        """Accumulate the elapsed seconds of one model when requested."""
        if timing is not None:
            timing[name] = timing.get(name, 0.0) + time.perf_counter() - mark
    x_all = arr['x']
    y_all = arr['y']
    uv_all = arr['uv']

    te_x = x_all[year_start:year_end]
    tr_x = x_all[train_index]
    tr_y = y_all[train_index]
    tr_uv = uv_all[train_index]

    n_sample, n_hour = te_x.shape[0], te_x.shape[1]
    shape = (n_sample, n_hour, 2)
    result = {name: np.full(shape, np.nan, np.float32) for name in selected}

    te_mask = ~np.isnan(te_x).any(axis=2)
    ec_mask = (~np.isnan(te_x[..., 3])) & (~np.isnan(te_x[..., 4]))

    mark = time.perf_counter()
    if 'ECMWF-IFS' in selected and ec_mask.any():
        result['ECMWF-IFS'][ec_mask] = uv_to_ds(te_x[ec_mask][:, 3:5], marker)
    _tick('ECMWF-IFS', mark)

    # Each (sample, lead time) pair is one training row, so a missing slot
    # no longer discards the whole sample.
    fx_flat = tr_x.reshape(-1, tr_x.shape[-1])
    sc_flat = tr_y.reshape(-1, tr_y.shape[-1])
    uv_flat = tr_uv.reshape(-1, tr_uv.shape[-1])
    keep_fx = ~np.isnan(fx_flat).any(axis=1)
    keep_sc = keep_fx & ~np.isnan(sc_flat).any(axis=1)
    keep_uv = keep_fx & ~np.isnan(uv_flat).any(axis=1)
    if keep_sc.sum() < 100 or not te_mask.any():
        return result

    fx_tr = fx_flat[keep_sc]
    sc_tr = sc_flat[keep_sc]
    uv_tr = uv_flat[keep_uv]
    fx_uv = fx_flat[keep_uv]
    fx_te = te_x[te_mask]

    mark = time.perf_counter()
    if 'PDFM' in selected:
        pred_uv = np.full((fx_te.shape[0], 2), np.nan, np.float32)
        for k in range(2):
            obs_q, fcst_q = quantile_map(uv_tr[:, k], fx_tr[:, 3 + k])
            pred_uv[:, k] = np.interp(fx_te[:, 3 + k], fcst_q, obs_q)
        result['PDFM'][te_mask] = uv_to_ds(pred_uv, marker)
    _tick('PDFM', mark)

    mark = time.perf_counter()
    if 'MLR' in selected:
        mlr_models = fit_mlr(fx_tr, sc_tr)
        result['MLR'][te_mask] = sc_to_ds(
            predict_multi(mlr_models, fx_te), marker
        )
    _tick('MLR', mark)

    mark = time.perf_counter()
    if 'LightGBM-A' in selected:
        sc_models = fit_lgb(fx_tr, sc_tr, cfg)
        result['LightGBM-A'][te_mask] = sc_to_ds(
            predict_multi(sc_models, fx_te), marker
        )
    _tick('LightGBM-A', mark)

    mark = time.perf_counter()
    if 'LightGBM-B' in selected:
        uv_models = fit_lgb(fx_uv, uv_tr, cfg)
        result['LightGBM-B'][te_mask] = uv_to_ds(
            predict_multi(uv_models, fx_te), marker
        )
    _tick('LightGBM-B', mark)
    return result


def _with_jobs(cfg: dict, n_jobs: int) -> dict:
    """
    Return a copy of the configuration with a fixed LightGBM thread count.

    Args:
        cfg: Configuration dictionary.
        n_jobs: Threads used inside one worker.

    Returns:
        dict: Copied configuration.
    """
    cfg_one = dict(cfg)
    cfg_one['model'] = dict(cfg['model'])
    cfg_one['model']['lgb_n_jobs'] = int(n_jobs)
    return cfg_one


def _merge_timing(total: dict, part: dict) -> None:
    """
    Add one worker's per-model seconds into the fold total.

    Args:
        total: Accumulated values, modified in place.
        part: Values returned by one worker.
    """
    for key, value in part.items():
        total[key] = total.get(key, 0) + value


def _station_task(task: tuple) -> dict:
    """
    Process one station chunk and write predictions into shared memmaps.

    Args:
        task: (stations, cfg, start, end, train_index, selected,
            pred_paths, block_start).

    Returns:
        dict: Seconds per model plus the processed station count.
    """
    (station_ids, cfg, start, end, train_index,
     selected, pred_paths, block_start) = task
    maps = {
        name: np.load(pred_paths[name], mmap_mode='r+')
        for name in selected
    }
    timing = {}
    done = 0
    for offset, sta in enumerate(station_ids):
        arrays = workflow.load_arrays(sta, cfg)
        if arrays is None:
            continue
        result = process_station(
            arrays, cfg, start, end, train_index, selected, timing
        )
        for name in selected:
            if name in result:
                maps[name][:, :, block_start + offset, :] = result[name]
        done += 1
    for handle in maps.values():
        handle.flush()
    timing['stations'] = done
    return timing


def _write_timing(pred_dir: pathlib.Path, rows: list) -> None:
    """
    Write the per-model training cost table next to the predictions.

    Args:
        pred_dir: Prediction directory.
        rows: Timing rows collected across folds.
    """
    if not rows:
        return
    log_dir = pred_dir.parent / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / 'train_timing.csv'
    fields = ['model', 'year', 'stations', 'seconds', 'sec_per_station']
    with open(path, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print('Timing: {0}'.format(path), flush=True)


def main(
    limit: int = 0,
    folds: int = 0,
    models: typing.Optional[typing.Sequence[str]] = None,
    years_wanted: typing.Optional[typing.Sequence[int]] = None,
    cores: int = 0,
) -> None:
    """
    Train and predict all folds.

    Args:
        limit: Number of stations to process (0 means all).
        folds: Number of folds to run (0 means all).
        models: Model names to run (None means all).
        years_wanted: Held-out years to run (None means all folds).
        cores: Total thread budget (0 uses every logical CPU).
    """
    cfg = workflow.load_config()
    station_ids = workflow.station_ids(cfg)
    if limit:
        station_ids = station_ids[:limit]
    n_station = len(station_ids)
    if n_station == 0:
        print('No station found', flush=True)
        return

    slices = workflow.year_slices(cfg)
    years = sorted(slices.keys())
    selected = tuple(models) if models else MODELS
    unknown = [name for name in selected if name not in MODELS]
    if unknown:
        print(
            'Unknown model: {0}; valid: {1}'.format(
                ', '.join(unknown), ', '.join(MODELS)
            ),
            flush=True,
        )
        return
    pred_dir = workflow.pred_root(cfg)
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(
        'Stations: {0}  folds: {1}  models: {2}'.format(
            n_station, len(years), ', '.join(selected)
        ),
        flush=True,
    )

    n_hour = int(workflow.target_hours(cfg).size)
    n_cpu = os.cpu_count() or 4
    budget = cores if cores > 0 else n_cpu
    n_worker = max(1, min(n_station, max(1, budget // WORKER_THREADS)))
    cfg_worker = _with_jobs(cfg, max(2, budget // n_worker))
    print(
        'Workers: {0}  threads each: {1}'.format(
            n_worker, cfg_worker['model']['lgb_n_jobs']
        ),
        flush=True,
    )
    timing_rows = list()
    for fold_index, year in enumerate(years):
        if folds and fold_index >= folds:
            break
        if years_wanted and year not in years_wanted:
            continue
        start, end = slices[year]
        train_index = np.concatenate([
            np.arange(*slices[other]) for other in years if other != year
        ])
        n_year = end - start
        shape = (n_year, n_hour, n_station, 2)
        pred_paths = dict()
        for name in selected:
            path = pred_dir / '{0}_{1}.npy'.format(FILE_NAMES[name], year)
            pred_paths[name] = str(path)
            block = np.lib.format.open_memmap(
                path, mode='w+', dtype=np.float32, shape=shape
            )
            block[:] = np.nan
            block.flush()
            del block
        fold_start = time.time()

        step = max(1, (n_station + n_worker - 1) // n_worker)
        chunks = [
            (station_ids[begin:begin + step], begin)
            for begin in range(0, n_station, step)
        ]
        tasks = [
            (chunk, cfg_worker, start, end, train_index, selected,
             pred_paths, block_start)
            for chunk, block_start in chunks
        ]
        timing = dict()
        if n_worker <= 1:
            for task in tasks:
                _merge_timing(timing, _station_task(task))
        else:
            finished = 0
            with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
                for part in pool.map(_station_task, tasks):
                    _merge_timing(timing, part)
                    finished += 1
                    print(
                        '  fold {0} chunk {1}/{2}'.format(
                            year, finished, len(tasks)
                        ),
                        flush=True,
                    )
        elapsed = (time.time() - fold_start) / 60
        done = int(timing.get('stations', 0))
        for name in selected:
            seconds = timing.get(name, 0.0)
            timing_rows.append({
                'model': name,
                'year': year,
                'stations': done,
                'seconds': round(seconds, 2),
                'sec_per_station': (
                    round(seconds / done, 4) if done else 0.0
                ),
            })
        print(
            'Fold {0} finished in {1:.1f} min ({2} stations)'.format(
                year, elapsed, done
            ),
            flush=True,
        )
    _write_timing(pred_dir, timing_rows)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Train station correction models'
    )
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--folds', type=int, default=0)
    parser.add_argument('--models', type=str, default='')
    parser.add_argument(
        '--years', type=str, default='',
        help='Comma-separated held-out years (empty means every fold)'
    )
    parser.add_argument(
        '--cores', type=int, default=0,
        help='Total thread budget (0 uses every logical CPU)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    print('Program train.py started', flush=True)
    args = parse_args()
    model_list = [
        item.strip() for item in args.models.split(',') if item.strip()
    ]
    year_list = [
        int(item.strip()) for item in args.years.split(',') if item.strip()
    ]
    main(
        limit=args.limit,
        folds=args.folds,
        models=model_list or None,
        years_wanted=year_list or None,
        cores=args.cores,
    )

