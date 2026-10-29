#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the 2411-station correction models with the per-station scheme.

For every leave-one-year-out fold and every station, fit five models:
the raw ECMWF-IFS wind (no training), a quantile-mapping PDFM baseline,
a multiple linear regression baseline, the angular-encoding LightGBM and
the vector-decomposition LightGBM. Predictions are written as
(direction, speed) arrays with shape (n_samples, 24, n_stations, 2).

Founded in 2026-10-29
Modified in 2026-10-29
@author: yinlb, deepseek-flash
"""

import argparse
import pathlib
import time
import typing

import lightgbm as lgbm
import numpy as np
from sklearn import linear_model as lm

from src import data_2411

MODELS = ('ECMWF-IFS', 'PDFM', 'MLR', 'LightGBM-A', 'LightGBM-B')
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

    Returns:
        typing.Dict[str, np.ndarray]: Predictions as (direction, speed)
        arrays with shape (n_samples, n_hours, 2).
    """
    marker = float(cfg['obs']['calm_marker'])
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

    if 'ECMWF-IFS' in selected and ec_mask.any():
        result['ECMWF-IFS'][ec_mask] = uv_to_ds(te_x[ec_mask][:, 3:5], marker)

    # Calm samples carry NaN in the cosine slot, so they are excluded here.
    tr_mask = (~np.isnan(tr_x).any(axis=2)) & (~np.isnan(tr_y).any(axis=2))
    if tr_mask.sum() < 100 or not te_mask.any():
        return result

    fx_tr = tr_x[tr_mask]
    sc_tr = tr_y[tr_mask]
    uv_tr = tr_uv[tr_mask]
    fx_te = te_x[te_mask]

    if 'PDFM' in selected:
        pred_uv = np.full((fx_te.shape[0], 2), np.nan, np.float32)
        for k in range(2):
            obs_q, fcst_q = quantile_map(uv_tr[:, k], fx_tr[:, 3 + k])
            pred_uv[:, k] = np.interp(fx_te[:, 3 + k], fcst_q, obs_q)
        result['PDFM'][te_mask] = uv_to_ds(pred_uv, marker)

    if 'MLR' in selected:
        mlr_models = fit_mlr(fx_tr, sc_tr)
        result['MLR'][te_mask] = sc_to_ds(
            predict_multi(mlr_models, fx_te), marker
        )

    if 'LightGBM-A' in selected:
        sc_models = fit_lgb(fx_tr, sc_tr, cfg)
        result['LightGBM-A'][te_mask] = sc_to_ds(
            predict_multi(sc_models, fx_te), marker
        )

    if 'LightGBM-B' in selected:
        uv_models = fit_lgb(fx_tr, uv_tr, cfg)
        result['LightGBM-B'][te_mask] = uv_to_ds(
            predict_multi(uv_models, fx_te), marker
        )
    return result


def main(
    limit: int = 0,
    folds: int = 0,
    models: typing.Optional[typing.Sequence[str]] = None,
) -> None:
    """
    Train and predict all folds.

    Args:
        limit: Number of stations to process (0 means all).
        folds: Number of folds to run (0 means all).
        models: Model names to run (None means all).
    """
    cfg = data_2411.load_config()
    station_ids = data_2411.station_ids(cfg)
    if limit:
        station_ids = station_ids[:limit]
    n_station = len(station_ids)
    if n_station == 0:
        print('No station found', flush=True)
        return

    slices = data_2411.year_slices(cfg)
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
    pred_dir = pathlib.Path(cfg['paths']['pred_dir'])
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(
        'Stations: {0}  folds: {1}  models: {2}'.format(
            n_station, len(years), ', '.join(selected)
        ),
        flush=True,
    )

    for fold_index, year in enumerate(years):
        if folds and fold_index >= folds:
            break
        start, end = slices[year]
        train_index = np.concatenate([
            np.arange(*slices[other]) for other in years if other != year
        ])
        n_year = end - start
        output = {
            name: np.full((n_year, 24, n_station, 2), np.nan, np.float32)
            for name in selected
        }
        fold_start = time.time()

        for index, sta in enumerate(station_ids, 1):
            arr = data_2411.load_cache(sta, cfg)
            if arr is None:
                continue
            result = process_station(
                arr, cfg, start, end, train_index, selected
            )
            for name in selected:
                if name in result:
                    output[name][:, :, index - 1, :] = result[name]

            if index % 20 == 0 or index == n_station:
                elapsed = (time.time() - fold_start) / 60
                rate = index / elapsed if elapsed > 0 else 0.0
                remain = (n_station - index) / rate if rate > 0 else 0.0
                print(
                    '  fold {0} [{1}/{2}]  {3:.1f} min  eta {4:.1f} min'.format(
                        year, index, n_station, elapsed, remain
                    ),
                    flush=True,
                )

        for name in selected:
            path = pred_dir / '{0}_{1}.npy'.format(FILE_NAMES[name], year)
            np.save(path, output[name])
        print(
            'Fold {0} finished in {1:.1f} min'.format(
                year, (time.time() - fold_start) / 60
            ),
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Train 2411-station correction models'
    )
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--folds', type=int, default=0)
    parser.add_argument('--models', type=str, default='')
    return parser.parse_args()


if __name__ == '__main__':
    print('Program train_2411.py started', flush=True)
    args = parse_args()
    model_list = [
        item.strip() for item in args.models.split(',') if item.strip()
    ]
    main(limit=args.limit, folds=args.folds, models=model_list or None)

