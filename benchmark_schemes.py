#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare the four modelling schemes of the paper (tables 6 and 7).

Scheme 1 (unified) trains one model on every station and lead time, scheme 2
(by lead time) one model per lead time, scheme 3 (by station) one model per
station, and scheme 4 (by station and lead time) one model per station and
lead time. Schemes 1 and 2 stream the full feature matrix through
``lightgbm.Sequence`` and retain only the one-dimensional labels.

Founded in 2026-11-04
Modified in 2026-11-20
@author: yinlb
"""

import argparse
import os
import pathlib
import time
import typing
from concurrent.futures import ProcessPoolExecutor

import lightgbm as lgbm
import numpy as np
import pandas as pd

from src import data_2411
from train_2411 import fit_lgb, process_station, sc_to_ds
from verify_2411 import ScoreAcc, write_text

SCHEMES = ('unified', 'lead', 'station', 'station_lead')
PROGRESS_EVERY = 100


class CacheSequence(lgbm.Sequence):
    """
    Stream feature rows from per-station caches and retain only the labels.

    Args:
        station_ids: Stations to stream.
        cfg: Configuration dictionary.
        rows: Sample indices of the training years.
        slots: Lead-time slots to keep.
        target: Target column index in the (speed, sin, cos) array.
    """

    def __init__(
        self,
        station_ids: typing.Sequence[str],
        cfg: dict,
        rows: np.ndarray,
        slots: typing.Sequence[int],
        target: int,
    ) -> None:
        self.station_ids = list(station_ids)
        self.cfg = cfg
        self.rows = rows
        self.slots = list(slots)
        self.target = target
        self.n_feature = len(data_2411.factor_columns(cfg)[0])
        self.offsets = np.zeros(len(station_ids) + 1, dtype=np.int64)
        self._cache = {}
        self._order = []
        self._cache_size = 4
        labels = list()
        for index in range(len(station_ids)):
            _, target_val = self._station_data(index)
            labels.append(target_val)
            self.offsets[index + 1] = self.offsets[index] + len(target_val)
        if labels and self.offsets[-1] > 0:
            self.labels = np.concatenate(labels).astype(np.float32, copy=False)
        else:
            self.labels = np.empty(0, dtype=np.float32)

    def __len__(self) -> int:
        """Return the total number of valid feature rows."""
        return int(self.offsets[-1])

    def _station_data(
        self, index: int
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Load one station, keeping a few recent blocks in memory."""
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        arr = data_2411.load_arrays(self.station_ids[index], self.cfg)
        if arr is None:
            x_val = np.empty((0, self.n_feature), dtype=np.float32)
            target_val = np.empty(0, dtype=np.float32)
        else:
            x_all = arr['x'][self.rows][:, self.slots]
            y_all = arr['y'][self.rows][:, self.slots]
            x_val = x_all.reshape(-1, x_all.shape[-1])
            y_val = y_all.reshape(-1, y_all.shape[-1])
            keep = (
                (~np.isnan(x_val).any(axis=1))
                & (~np.isnan(y_val).any(axis=1))
            )
            x_val = x_val[keep]
            target_val = y_val[keep, self.target]
        self._cache[index] = (x_val, target_val)
        self._order.append(index)
        if len(self._order) > self._cache_size:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        return x_val, target_val

    def _get_one(self, index: int) -> np.ndarray:
        """Return one feature row by its global valid-row index."""
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError('Sequence index out of range')
        station = int(np.searchsorted(self.offsets, index, side='right') - 1)
        local = index - int(self.offsets[station])
        x_val, _ = self._station_data(station)
        # LightGBM's Sequence sampling path requires double precision.
        return x_val[local].astype(np.float64, copy=False)

    def __getitem__(self, index):
        """Return one row or a two-dimensional batch of feature rows."""
        if isinstance(index, (int, np.integer)):
            return self._get_one(int(index))
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
        elif isinstance(index, (list, np.ndarray)):
            indices = index
        else:
            raise TypeError(
                'Sequence index must be integer, slice or list, got {0}'
                .format(type(index).__name__)
            )
        rows = [self._get_one(int(item)) for item in indices]
        if not rows:
            return np.empty((0, self.n_feature), dtype=np.float64)
        return np.stack(rows)


def parallel_budget(cfg: dict) -> int:
    """
    Return the total thread budget shared by the four schemes.

    Args:
        cfg: Configuration dictionary.

    Returns:
        int: Inner threads x outer workers, close to the core count.
    """
    fixed = int(cfg['parallel']['n_total'])
    if fixed > 0:
        return fixed
    cores = os.cpu_count() or 4
    reserve = int(cfg['parallel']['reserve'])
    return max(4, (cores - reserve) // 4 * 4)


def _split(items: typing.Sequence, parts: int) -> typing.List[list]:
    """
    Split a sequence into roughly equal contiguous chunks.

    Args:
        items: Sequence to split.
        parts: Number of chunks.

    Returns:
        typing.List[list]: Chunks in the original order.
    """
    parts = max(1, int(parts))
    step = max(1, (len(items) + parts - 1) // parts)
    return [list(items[i:i + step]) for i in range(0, len(items), step)]


def _with_jobs(cfg: dict, n_jobs: int) -> dict:
    """
    Return a copy of the configuration with a fixed thread count.

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


def lgb_params(cfg: dict) -> dict:
    """
    Return the LightGBM parameters shared by every scheme.

    Args:
        cfg: Configuration dictionary.

    Returns:
        dict: Parameter dictionary.
    """
    params = dict(cfg['model'].get('lgb_params') or {})
    params.update(
        n_jobs=cfg['model']['lgb_n_jobs'],
        bin_construct_sample_cnt=int(cfg['model']['bin_sample_cnt']),
        verbose=-1,
        force_col_wise=True,
        random_state=cfg['model']['random_seed'],
    )
    return params


def fit_frames(
    station_ids: typing.Sequence[str],
    cfg: dict,
    rows: np.ndarray,
    slots: typing.Sequence[int],
) -> typing.List[lgbm.Booster]:
    """
    Fit one streaming model per angular-encoding target.

    Args:
        station_ids: Training stations.
        cfg: Configuration dictionary.
        rows: Sample indices of the training years.
        slots: Lead-time slots to keep.

    Returns:
        typing.List[lgbm.Booster]: Speed, sine and cosine models.
    """
    params = lgb_params(cfg)
    models = list()
    for target in range(3):
        sequence = CacheSequence(station_ids, cfg, rows, slots, target)
        dataset = lgbm.Dataset(
            sequence, label=sequence.labels, params=params,
            free_raw_data=False,
        )
        models.append(lgbm.train(params, dataset))
    return models


def predict_sc(
    models: typing.Sequence[typing.Any],
    x_val: np.ndarray,
    marker: float,
) -> np.ndarray:
    """
    Predict angular targets on rows with complete features.

    Args:
        models: Speed, sine and cosine models.
        x_val: Features with shape (n_samples, n_features).
        marker: Calm-wind direction marker.

    Returns:
        np.ndarray: Direction and speed, with NaN on incomplete feature rows.
    """
    pred = np.full((x_val.shape[0], 2), np.nan, dtype=np.float32)
    valid = ~np.isnan(x_val).any(axis=1)
    if valid.any():
        columns = [model.predict(x_val[valid]) for model in models]
        pred[valid] = sc_to_ds(np.stack(columns, axis=-1), marker)
    return pred


def score_frames(
    models: typing.List[lgbm.Booster],
    station_ids: typing.Sequence[str],
    cfg: dict,
    rows: np.ndarray,
    slots: typing.Sequence[int],
    marker: float,
) -> ScoreAcc:
    """
    Score streaming models on one fold without storing the predictions.

    Args:
        models: Speed, sine and cosine models.
        station_ids: Evaluation stations.
        cfg: Configuration dictionary.
        rows: Sample indices of the held-out year.
        slots: Lead-time slots to score.
        marker: Calm-wind direction marker.

    Returns:
        ScoreAcc: Accumulated scores and sample counts.
    """
    acc = ScoreAcc(marker)
    for index, sta in enumerate(station_ids, 1):
        arr = data_2411.load_arrays(sta, cfg)
        x_all = arr['x'][rows][:, slots]
        x_val = x_all.reshape(-1, x_all.shape[-1])
        pred = predict_sc(models, x_val, marker)
        acc.add(
            arr['dir'][rows][:, slots].reshape(-1),
            arr['spd'][rows][:, slots].reshape(-1),
            pred[..., 0], pred[..., 1],
        )
        if index % PROGRESS_EVERY == 0 or index == len(station_ids):
            print('    scored {0}/{1}'.format(index, len(station_ids)),
                  flush=True)
    return acc



def fold_rows(
    cfg: dict, year: int
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Return the training and held-out sample indices of one fold.

    Args:
        cfg: Configuration dictionary.
        year: Held-out year.

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: Training and test indices.
    """
    slices = data_2411.year_slices(cfg)
    if year not in slices:
        raise ValueError('Unknown year: {0}'.format(year))
    test = np.arange(slices[year][0], slices[year][1])
    train = np.concatenate([
        np.arange(*slices[other])
        for other in sorted(slices) if other != year
    ])
    return train, test


def run_unified(
    station_ids: typing.Sequence[str], cfg: dict, year: int, marker: float,
    n_jobs: int,
) -> ScoreAcc:
    """
    Run scheme 1: one model for every station and lead time.

    Args:
        station_ids: Stations of the fold.
        cfg: Configuration dictionary.
        year: Held-out year.
        marker: Calm-wind direction marker.
        n_jobs: Threads used by LightGBM.

    Returns:
        ScoreAcc: Accumulated scores of the fold.
    """
    train, test = fold_rows(cfg, year)
    slots = list(range(len(data_2411.target_hours(cfg))))
    cfg_one = _with_jobs(cfg, n_jobs)
    models = fit_frames(station_ids, cfg_one, train, slots)
    return score_frames(models, station_ids, cfg_one, test, slots, marker)


def _lead_task(task: tuple) -> tuple:
    """
    Fit and score the lead-time models of one slot chunk.

    Args:
        task: (stations, cfg, year, slots, marker, n_jobs).

    Returns:
        tuple: Overall accumulator and {slot: accumulator}.
    """
    station_ids, cfg, year, slots, marker, n_jobs = task
    train, test = fold_rows(cfg, year)
    cfg_one = _with_jobs(cfg, n_jobs)
    models = {
        slot: fit_frames(station_ids, cfg_one, train, [slot])
        for slot in slots
    }
    total = ScoreAcc(marker)
    per_slot = {}
    for sta in station_ids:
        arr = data_2411.load_arrays(sta, cfg)
        for slot in slots:
            x_val = arr['x'][test][:, slot]
            pred = predict_sc(models[slot], x_val, marker)
            args = (
                arr['dir'][test][:, slot].reshape(-1),
                arr['spd'][test][:, slot].reshape(-1),
                pred[..., 0], pred[..., 1],
            )
            if slot not in per_slot:
                per_slot[slot] = ScoreAcc(marker)
            per_slot[slot].add(*args)
            total.add(*args)
    return total, per_slot


def _station_task(task: tuple) -> ScoreAcc:
    """
    Run scheme 3 on one chunk of stations.

    Args:
        task: (stations, cfg, year, marker, n_jobs).

    Returns:
        ScoreAcc: Accumulator of the chunk.
    """
    station_ids, cfg, year, marker, n_jobs = task
    train, test = fold_rows(cfg, year)
    cfg_one = _with_jobs(cfg, n_jobs)
    total = ScoreAcc(marker)
    for sta in station_ids:
        arr = data_2411.load_arrays(sta, cfg)
        result = process_station(
            arr, cfg_one, test[0], test[-1] + 1, train, ('LightGBM-A',)
        )
        pred = result.get('LightGBM-A')
        if pred is None:
            continue
        total.add(
            arr['dir'][test], arr['spd'][test],
            pred[..., 0], pred[..., 1],
        )
    return total


def _station_lead_task(task: tuple) -> tuple:
    """
    Run scheme 4 on one chunk of stations, one thread per model.

    Args:
        task: (stations, cfg, year, marker).

    Returns:
        tuple: Overall accumulator and per-slot accumulators.
    """
    station_ids, cfg, year, marker = task
    train, test = fold_rows(cfg, year)
    cfg_one = _with_jobs(cfg, 1)
    n_slot = len(data_2411.target_hours(cfg))
    total = ScoreAcc(marker)
    slot_acc = [ScoreAcc(marker) for _ in range(n_slot)]
    for sta in station_ids:
        arr = data_2411.load_arrays(sta, cfg)
        for slot in range(n_slot):
            x_tr = arr['x'][train][:, slot]
            y_tr = arr['y'][train][:, slot]
            keep = (
                (~np.isnan(x_tr).any(axis=1))
                & (~np.isnan(y_tr).any(axis=1))
            )
            if keep.sum() < 100:
                continue
            models = fit_lgb(x_tr[keep], y_tr[keep], cfg_one)
            x_te = arr['x'][test][:, slot]
            pred = predict_sc(models, x_te, marker)
            args = (
                arr['dir'][test][:, slot].reshape(-1),
                arr['spd'][test][:, slot].reshape(-1),
                pred[..., 0], pred[..., 1],
            )
            slot_acc[slot].add(*args)
            total.add(*args)
    return total, slot_acc


def _available_gb() -> float:
    """
    Return the currently available RAM in GB.

    Returns:
        float: Available memory, or -1.0 when psutil is unavailable.
    """
    try:
        import psutil
    except ImportError:
        return -1.0
    return psutil.virtual_memory().available / 1024 ** 3


def _preload_need_gb(
    n_station: int, n_sample: int, n_slot: int
) -> float:
    """
    Estimate the RAM needed by the preloaded matrix and its Dataset.

    Per row the estimate counts 26 float32 features (104 B), the label, uv
    and slot buffers (20 B), the binned Dataset (26 B) and the gradients (8 B).

    Args:
        n_station: Number of stations.
        n_sample: Training samples per station.
        n_slot: Lead-time slots.

    Returns:
        float: Upper-bound estimate in GB.
    """
    rows = float(n_station) * float(n_sample) * float(n_slot)
    return rows * 158.0 / 1024 ** 3


def decide_stream(
    cfg: dict, n_station: int, n_sample: int, n_slot: int
) -> typing.Tuple[bool, float, float]:
    """
    Decide whether the training rows must be streamed.

    Args:
        cfg: Configuration dictionary.
        n_station: Number of stations.
        n_sample: Training samples per station.
        n_slot: Lead-time slots.

    Returns:
        typing.Tuple[bool, float, float]: Use stream, available GB, need GB.
    """
    mode = str(cfg['model']['stream']).lower()
    avail = _available_gb()
    need = _preload_need_gb(n_station, n_sample, n_slot)
    if mode == 'true':
        return True, avail, need
    if mode == 'false':
        return False, avail, need
    if avail < 0:
        return True, avail, need
    return avail < need * 1.3, avail, need


def _score_slots(
    models: dict, station_ids: typing.Sequence[str], cfg: dict,
    rows: np.ndarray, slots: typing.Sequence[int], marker: float,
) -> typing.Tuple[ScoreAcc, dict]:
    """
    Score the given slots with already fitted per-slot models.

    Args:
        models: Booster list keyed by slot index.
        station_ids: Stations of the fold.
        cfg: Configuration dictionary.
        rows: Sample indices of the held-out year.
        slots: Slots to score.
        marker: Calm-wind direction marker.

    Returns:
        typing.Tuple[ScoreAcc, dict]: Overall and per-slot accumulators.
    """
    acc = ScoreAcc(marker)
    per_slot = {slot: ScoreAcc(marker) for slot in slots}
    for sta in station_ids:
        arr = data_2411.load_arrays(sta, cfg)
        for slot in slots:
            x_val = arr['x'][rows][:, slot]
            pred = predict_sc(models[slot], x_val, marker)
            args = (
                arr['dir'][rows][:, slot].reshape(-1),
                arr['spd'][rows][:, slot].reshape(-1),
                pred[..., 0], pred[..., 1],
            )
            per_slot[slot].add(*args)
            acc.add(*args)
    return acc, per_slot


def load_matrix(
    station_ids: typing.Sequence[str], cfg: dict, rows: np.ndarray,
    slots: typing.Sequence[int],
) -> dict:
    """
    Load every training row into one in-memory matrix.

    Rows are ordered station -> sample -> slot. The slot index of every row is
    kept so that the lead-time scheme can slice its own subset.

    Args:
        station_ids: Training stations.
        cfg: Configuration dictionary.
        rows: Sample indices of the training years.
        slots: Lead-time slots to keep.

    Returns:
        dict: 'x', 'y', 'uv', 'slot' and the per-station 'offsets'.
    """
    n_feature = len(data_2411.factor_columns(cfg)[0])
    n_slot = len(slots)
    n_station = len(station_ids)
    counts = np.zeros(n_station, dtype=np.int64)
    for index, sta in enumerate(station_ids):
        arr = data_2411.load_arrays(sta, cfg)
        if arr is None:
            continue
        x_all = arr['x'][rows][:, slots]
        y_all = arr['y'][rows][:, slots]
        x_flat = x_all.reshape(-1, x_all.shape[-1])
        y_flat = y_all.reshape(-1, y_all.shape[-1])
        keep = (
            (~np.isnan(x_flat).any(axis=1))
            & (~np.isnan(y_flat).any(axis=1))
        )
        counts[index] = int(keep.sum())
    offsets = np.zeros(n_station + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    total = int(offsets[-1])
    print(
        '  preloading {0} rows ({1:.1f} GB)'.format(
            total, total * (n_feature + 5) * 4 / 1024 ** 3
        ),
        flush=True,
    )
    x_buf = np.empty((total, n_feature), dtype=np.float32)
    y_buf = np.empty((total, 3), dtype=np.float32)
    uv_buf = np.empty((total, 2), dtype=np.float32)
    slot_buf = np.empty(total, dtype=np.int16)
    for index, sta in enumerate(station_ids):
        arr = data_2411.load_arrays(sta, cfg)
        if arr is None:
            continue
        x_all = arr['x'][rows][:, slots]
        y_all = arr['y'][rows][:, slots]
        uv_all = arr['uv'][rows][:, slots]
        x_flat = x_all.reshape(-1, x_all.shape[-1])
        y_flat = y_all.reshape(-1, y_all.shape[-1])
        uv_flat = uv_all.reshape(-1, uv_all.shape[-1])
        keep = (
            (~np.isnan(x_flat).any(axis=1))
            & (~np.isnan(y_flat).any(axis=1))
        )
        slot_full = np.tile(np.arange(n_slot, dtype=np.int16), len(rows))
        block = slice(int(offsets[index]), int(offsets[index + 1]))
        x_buf[block] = x_flat[keep]
        y_buf[block] = y_flat[keep]
        uv_buf[block] = uv_flat[keep]
        slot_buf[block] = slot_full[keep]
    return {
        'x': x_buf, 'y': y_buf, 'uv': uv_buf,
        'slot': slot_buf, 'offsets': offsets,
    }


def run_unified_matrix(
    station_ids: typing.Sequence[str], cfg: dict, year: int, marker: float,
    n_total: int,
) -> ScoreAcc:
    """
    Run scheme 1 from a preloaded matrix instead of a Sequence.

    Args:
        station_ids: Stations of the fold.
        cfg: Configuration dictionary.
        year: Held-out year.
        marker: Calm-wind direction marker.
        n_total: Threads used by LightGBM.

    Returns:
        ScoreAcc: Accumulated scores of the fold.
    """
    train, test = fold_rows(cfg, year)
    slots = list(range(len(data_2411.target_hours(cfg))))
    matrix = load_matrix(station_ids, cfg, train, slots)
    cfg_one = _with_jobs(cfg, n_total)
    params = lgb_params(cfg_one)
    models = list()
    for target in range(3):
        dataset = lgbm.Dataset(
            matrix['x'], label=matrix['y'][:, target], params=params,
            free_raw_data=False,
        )
        models.append(lgbm.train(params, dataset))
    del matrix
    return score_frames(models, station_ids, cfg_one, test, slots, marker)


def run_lead_matrix(
    station_ids: typing.Sequence[str], cfg: dict, year: int, marker: float,
    n_total: int,
) -> typing.Tuple[ScoreAcc, list]:
    """
    Run scheme 2 from a preloaded matrix, one model per lead time.

    Args:
        station_ids: Stations of the fold.
        cfg: Configuration dictionary.
        year: Held-out year.
        marker: Calm-wind direction marker.
        n_total: Threads used by LightGBM.

    Returns:
        typing.Tuple[ScoreAcc, list]: Overall and per-slot accumulators.
    """
    train, test = fold_rows(cfg, year)
    n_slot = len(data_2411.target_hours(cfg))
    slots = list(range(n_slot))
    matrix = load_matrix(station_ids, cfg, train, slots)
    cfg_one = _with_jobs(cfg, n_total)
    params = lgb_params(cfg_one)
    total = ScoreAcc(marker)
    per_slot = list()
    for slot in slots:
        index = np.flatnonzero(matrix['slot'] == slot)
        if index.size < 100:
            per_slot.append(ScoreAcc(marker))
            continue
        models = list()
        for target in range(3):
            dataset = lgbm.Dataset(
                matrix['x'][index], label=matrix['y'][index, target],
                params=params, free_raw_data=False,
            )
            models.append(lgbm.train(params, dataset))
        slot_acc = _score_slots(
            {slot: models}, station_ids, cfg_one, test, [slot], marker
        )[1][slot]
        total.merge(slot_acc)
        per_slot.append(slot_acc)
    del matrix
    return total, per_slot


def run_lead_parallel(
    station_ids: typing.Sequence[str], cfg: dict, year: int, marker: float,
    n_total: int,
) -> typing.Tuple[ScoreAcc, list]:
    """
    Run scheme 2 with two workers over the lead-time slots.

    Args:
        station_ids: Stations of the fold.
        cfg: Configuration dictionary.
        year: Held-out year.
        marker: Calm-wind direction marker.
        n_total: Total thread budget.

    Returns:
        typing.Tuple[ScoreAcc, list]: Overall and per-slot accumulators.
    """
    outer = 2
    inner = max(2, n_total // outer)
    n_slot = len(data_2411.target_hours(cfg))
    chunks = _split(list(range(n_slot)), outer)
    tasks = [
        (station_ids, cfg, year, chunk, marker, inner) for chunk in chunks
    ]
    with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        results = list(pool.map(_lead_task, tasks))
    total = ScoreAcc(marker)
    per_slot = [None] * n_slot
    for acc, slot_results in results:
        total.merge(acc)
        for slot, slot_acc in slot_results.items():
            per_slot[slot] = slot_acc
    return total, per_slot


def run_station_parallel(
    station_ids: typing.Sequence[str], cfg: dict, year: int, marker: float,
    n_total: int,
) -> ScoreAcc:
    """
    Run scheme 3 with four workers over the stations.

    Args:
        station_ids: Stations of the fold.
        cfg: Configuration dictionary.
        year: Held-out year.
        marker: Calm-wind direction marker.
        n_total: Total thread budget.

    Returns:
        ScoreAcc: Accumulated scores of the fold.
    """
    outer = 4
    inner = max(2, n_total // outer)
    chunks = _split(station_ids, outer)
    tasks = [(chunk, cfg, year, marker, inner) for chunk in chunks]
    with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        results = list(pool.map(_station_task, tasks))
    total = ScoreAcc(marker)
    for acc in results:
        total.merge(acc)
    return total


def run_station_lead_parallel(
    station_ids: typing.Sequence[str], cfg: dict, year: int, marker: float,
    n_total: int,
) -> typing.Tuple[ScoreAcc, list]:
    """
    Run scheme 4 with one worker per thread, stations split between them.

    Args:
        station_ids: Stations of the fold.
        cfg: Configuration dictionary.
        year: Held-out year.
        marker: Calm-wind direction marker.
        n_total: Total thread budget.

    Returns:
        typing.Tuple[ScoreAcc, list]: Overall and per-slot accumulators.
    """
    chunks = _split(station_ids, n_total)
    tasks = [(chunk, cfg, year, marker) for chunk in chunks]
    n_slot = len(data_2411.target_hours(cfg))
    with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        results = list(pool.map(_station_lead_task, tasks))
    total = ScoreAcc(marker)
    per_slot = [ScoreAcc(marker) for _ in range(n_slot)]
    for acc, slot_results in results:
        total.merge(acc)
        for slot, slot_acc in enumerate(slot_results):
            per_slot[slot].merge(slot_acc)
    return total, per_slot


def main(
    schemes: typing.Sequence[str] = SCHEMES,
    limit: int = 0,
    years: typing.Optional[typing.Sequence[int]] = None,
    subset: int = 0,
    cores: int = 0,
    no_stream: bool = False,
) -> None:
    """
    Run the selected schemes and write tables 6 and 7.

    Args:
        schemes: Scheme names to run.
        limit: Number of stations to use (0 means all).
        years: Held-out years to run (None means all folds).
        subset: Shared station count when scheme 4 is selected (0 means all).
        cores: Override of the thread budget (0 uses the configuration).
        no_stream: Preload the matrix instead of streaming.
    """
    unknown = [name for name in schemes if name not in SCHEMES]
    if unknown:
        print(
            'Unknown scheme: {0}; valid: {1}'.format(
                ', '.join(unknown), ', '.join(SCHEMES)
            ),
            flush=True,
        )
        return
    cfg = data_2411.load_config()
    n_total = parallel_budget(cfg)
    if cores:
        n_total = int(cores)
    print(
        'Parallel budget: cores={0}  total={1}'.format(
            os.cpu_count(), n_total
        ),
        flush=True,
    )
    station_ids = data_2411.station_ids(cfg)
    if limit:
        station_ids = station_ids[:limit]
    if 'station_lead' in schemes and subset:
        station_ids = station_ids[:subset]
    slices = data_2411.year_slices(cfg)
    year_list = list(years) if years else sorted(slices)
    n_slot = len(data_2411.target_hours(cfg))
    n_train = int(cfg['constants']['n_samples']) - max(
        end - start for start, end in slices.values()
    )
    stream, avail_gb, need_gb = decide_stream(
        cfg, len(station_ids), n_train, n_slot
    )
    if no_stream:
        stream = False
    print(
        'Preload: need={0:.1f} GB  available={1:.1f} GB  stream={2}'.format(
            need_gb, avail_gb, stream
        ),
        flush=True,
    )
    marker = float(cfg['obs']['calm_marker'])
    result_dir = data_2411.result_root(cfg)
    table_dir = result_dir / 'tables'
    table_dir.mkdir(parents=True, exist_ok=True)

    rows = list()
    lead_rows = list()
    total_acc = {scheme: ScoreAcc(marker) for scheme in schemes}
    slot_totals = {scheme: {} for scheme in schemes}
    minutes = {scheme: 0.0 for scheme in schemes}
    for scheme in schemes:
        for year in year_list:
            begin = time.time()
            print(
                'Scheme {0}  year {1}  stations {2}'.format(
                    scheme, year, len(station_ids)
                ),
                flush=True,
            )
            per_slot = None
            if scheme == 'unified':
                if stream:
                    acc = run_unified(
                        station_ids, cfg, year, marker, n_total
                    )
                else:
                    acc = run_unified_matrix(
                        station_ids, cfg, year, marker, n_total
                    )
            elif scheme == 'lead':
                if stream:
                    acc, per_slot = run_lead_parallel(
                        station_ids, cfg, year, marker, n_total
                    )
                else:
                    acc, per_slot = run_lead_matrix(
                        station_ids, cfg, year, marker, n_total
                    )
            elif scheme == 'station':
                acc = run_station_parallel(
                    station_ids, cfg, year, marker, n_total
                )
            else:
                acc, per_slot = run_station_lead_parallel(
                    station_ids, cfg, year, marker, n_total
                )
            total_acc[scheme].merge(acc)
            elapsed = (time.time() - begin) / 60.0
            minutes[scheme] += elapsed
            scores = acc.result()
            scores['scheme'] = scheme
            scores['year'] = year
            scores['n_stations'] = len(station_ids)
            scores['minutes'] = round(elapsed, 1)
            rows.append(scores)
            if per_slot is not None:
                for slot, slot_acc in enumerate(per_slot, 1):
                    if slot not in slot_totals[scheme]:
                        slot_totals[scheme][slot] = ScoreAcc(marker)
                    slot_totals[scheme][slot].merge(slot_acc)
                    item = slot_acc.result()
                    item['scheme'] = scheme
                    item['year'] = year
                    item['lead_index'] = slot
                    lead_rows.append(item)
            print(
                '  done in {0:.1f} min'.format(scores['minutes']),
                flush=True,
            )

    table = pd.DataFrame(rows)
    table.to_csv(
        table_dir / 'table6_by_fold.csv', index=False,
        encoding='utf-8-sig'
    )
    write_text(
        table, table_dir, 'table6_by_fold',
        'Table 6 modelling schemes by fold', index=False,
    )
    overall_rows = list()
    for scheme, acc in total_acc.items():
        item = acc.result()
        item['scheme'] = scheme
        item['n_stations'] = len(station_ids)
        item['minutes'] = round(minutes[scheme], 1)
        overall_rows.append(item)
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(
        table_dir / 'table6_schemes.csv', index=False,
        encoding='utf-8-sig'
    )
    write_text(
        overall, table_dir, 'table6_schemes',
        'Table 6 modelling schemes (pooled folds)', index=False,
    )
    if lead_rows:
        lead_table = pd.DataFrame(lead_rows)
        lead_table.to_csv(
            table_dir / 'table7_by_lead.csv', index=False,
            encoding='utf-8-sig'
        )
        write_text(
            lead_table, table_dir, 'table7_by_lead',
            'Table 7 scores by fold and lead index', index=False,
        )
        pooled_lead_rows = list()
        for scheme, slots in slot_totals.items():
            for slot, acc in sorted(slots.items()):
                item = acc.result()
                item['scheme'] = scheme
                item['lead_index'] = slot
                pooled_lead_rows.append(item)
        pooled_lead = pd.DataFrame(pooled_lead_rows)
        pooled_lead.to_csv(
            table_dir / 'table7_overall.csv', index=False,
            encoding='utf-8-sig'
        )
        write_text(
            pooled_lead, table_dir, 'table7_overall',
            'Table 7 pooled scores by lead index', index=False,
        )
    meta = (
        'cores={0}  n_total={1}\n'.format(os.cpu_count(), n_total)
        + 'unified: inner={0} outer=1\n'.format(n_total)
        + 'lead: inner={0} outer=2\n'.format(max(2, n_total // 2))
        + 'station: inner={0} outer=4\n'.format(max(2, n_total // 4))
        + 'station_lead: inner=1 outer={0}\n'.format(n_total)
        + 'stations={0}  years={1}\n'.format(
            len(station_ids), ','.join(str(item) for item in year_list)
        )
        + 'stream={0}  need={1:.1f}GB  available={2:.1f}GB\n'.format(
            stream, need_gb, avail_gb
        )
    )
    (table_dir / 'table6_meta.txt').write_text(meta, encoding='utf-8')
    print('Result dir: {0}'.format(table_dir), flush=True)


if __name__ == '__main__':
    print('Program benchmark_schemes.py started', flush=True)
    parser = argparse.ArgumentParser(
        description='Compare the four modelling schemes'
    )
    parser.add_argument(
        '--schemes', type=str, default=','.join(SCHEMES),
        help='schemes to run, comma separated (default: all four)',
    )
    parser.add_argument(
        '--limit', type=int, default=0,
        help='number of leading stations (0 means all, the default)',
    )
    parser.add_argument(
        '--years', type=str, default='',
        help='held-out years, comma separated (empty means all six)',
    )
    parser.add_argument(
        '--subset', type=int, default=0,
        help='shared station count, only used with station_lead '
             '(0 means all, the default)',
    )
    parser.add_argument(
        '--cores', type=int, default=0,
        help='override the thread budget (0 uses the configuration)',
    )
    parser.add_argument(
        '--no-stream', action='store_true',
        help='preload the matrix instead of streaming (needs large RAM)',
    )
    args = parser.parse_args()
    main(
        schemes=[item.strip() for item in args.schemes.split(',') if item],
        limit=args.limit,
        years=[
            int(item) for item in args.years.split(',') if item.strip()
        ] or None,
        subset=args.subset,
        cores=args.cores,
        no_stream=args.no_stream,
    )

