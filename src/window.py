#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lead-time helpers for the 3-hourly deployment product.

The raw factor files store 52 lead-time slots: 3-hourly up to 72 h, then
6-hourly up to 240 h. This module builds the interpolation matrix that maps
those slots onto a uniform 3-hourly grid of 80 slots (3, 6, ..., 240 h) and
provides the observation alignment needed to train on the full window.

Founded in 2026-11-04
Modified in 2026-11-04
@author: yinlb
"""

import pathlib
import typing

import numpy as np
import pandas as pd

from src import workflow

# Uniform 3-hourly lead times of the deployment product.
TARGET_LEADS = np.arange(3.0, 241.0, 3.0)


def interp_matrix(cfg: dict) -> np.ndarray:
    """
    Build the linear interpolation matrix from the 52 raw slots to 80 slots.

    Lead times that coincide with a raw slot get the raw value (fraction 0),
    the 72-78 h step and the later 6-hourly steps are interpolated linearly.

    Args:
        cfg: Configuration dictionary.

    Returns:
        np.ndarray: Weight matrix with shape (80, 52).
    """
    src = workflow.lead_hours(cfg)
    weights = np.zeros((TARGET_LEADS.size, src.size), dtype=np.float32)
    for index, hour in enumerate(TARGET_LEADS):
        left = int(np.searchsorted(src, hour, side='right') - 1)
        left = min(max(left, 0), src.size - 2)
        frac = (hour - src[left]) / (src[left + 1] - src[left])
        weights[index, left] = 1.0 - frac
        weights[index, left + 1] = frac
    return weights


def read_raw_factors(sta: str, cfg: dict) -> np.ndarray:
    """
    Read the raw factor array of one station, keeping the valid columns.

    Precipitation is converted from metres per slot to mm/h with the slot
    length of each raw lead time, exactly as ``workflow.load_factors`` does;
    any other step would make the deployment features differ from the
    training features.

    Args:
        sta: Station ID.
        cfg: Configuration dictionary.

    Returns:
        np.ndarray: Factors with shape (n_samples, 52, 26), float32.
    """
    path = pathlib.Path(cfg['paths']['data_dir']) / 'data{0}.npy'.format(sta)
    cols, _ = workflow.factor_columns(cfg)
    raw = np.load(path, mmap_mode='r')
    block = np.asarray(raw[:, :, cols], dtype=np.float32)
    if cfg['features']['precip']:
        steps = workflow.time_steps(cfg).astype(np.float32)
        pos = cols.index(workflow.COL_PRECIP)
        precip = np.maximum(block[:, :, pos], 0.0)
        block[:, :, pos] = precip * 1000.0 / steps[None, :]
    return block


def expand_leads(
    factors: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """
    Interpolate the 52 slots onto the 80-slot grid.

    A sample whose factors are entirely missing stays missing, because the
    matrix product propagates NaN along the whole slot axis.

    Args:
        factors: Factors with shape (n_samples, 52, n_features).
        weights: Matrix returned by interp_matrix, shape (80, 52).

    Returns:
        np.ndarray: Factors with shape (n_samples, 80, n_features).
    """
    out = np.einsum('sl,nlf->nsf', weights, factors, optimize=True)
    return out.astype(np.float32)


def full_sample_times(cfg: dict) -> pd.DatetimeIndex:
    """
    Return the UTC valid times of every (cycle, 52-slot) pair, flattened.

    Args:
        cfg: Configuration dictionary.

    Returns:
        pandas.DatetimeIndex: Times with length n_samples * 52.
    """
    start = pd.Timestamp(cfg['constants']['start_time'], tz='UTC')
    n_sample = int(cfg['constants']['n_samples'])
    cycle = int(cfg['constants']['cycle_hours'])
    leads = workflow.lead_hours(cfg)
    offsets = np.arange(n_sample)[:, None] * cycle + leads[None, :]
    return start + pd.to_timedelta(offsets.ravel(), unit='h')


def full_observations(
    sta: str, cfg: dict
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Align the observations to all 52 lead-time slots.

    Args:
        sta: Station ID.
        cfg: Configuration dictionary.

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: Direction and speed with shape
        (n_samples, 52). Both are None when the station has no observation.
    """
    frame = workflow._read_obs_frame(sta, cfg)
    if frame is None or frame.empty:
        return None, None
    dir_val, spd_val = workflow.clean_wind(
        frame['WIN_D_Avg_10mi'].values,
        frame['WIN_S_Avg_10mi'].values,
        cfg,
    )
    times = full_sample_times(cfg)
    shape = (int(cfg['constants']['n_samples']), len(workflow.lead_hours(cfg)))
    dir_out = pd.Series(dir_val, index=frame.index).reindex(times)
    spd_out = pd.Series(spd_val, index=frame.index).reindex(times)
    return (
        dir_out.to_numpy().reshape(shape),
        spd_out.to_numpy().reshape(shape),
    )
