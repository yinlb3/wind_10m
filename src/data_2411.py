#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data access layer for the 2411-station experiment.

Reads the per-station NWP factor arrays (data{sta}.npy) and the per-station
hourly observation CSVs (sta/{sta}.csv, plus sta/2024/{sta}.csv), cleans the
observations, interpolates the 3-hourly factors to hourly values inside the
evaluation window and encodes wind direction as (speed, sin, cos).

Founded in 2026-10-29
Modified in 2026-10-29
@author: yinlb, deepseek-flash
"""

import pathlib
import typing

import arrow
import numpy as np
import pandas as pd
import yaml

CONFIG_NAME = 'config.2411.yaml'
LOCAL_NAME = 'config.2411.local.yaml'

# Column layout of data{sta}.npy (31 columns).
COL_SURFACE = (0, 1, 2, 3, 4)
COL_PRECIP = 5
COL_LEVEL = {
    'T': (6, 7, 8, 9),
    'GH': (11, 12, 13, 14),
    'RH': (16, 17, 18, 19),
    'U': (21, 22, 23, 24),
    'V': (26, 27, 28, 29),
}
COL_LEVEL_ORDER = ('T', 'GH', 'RH', 'U', 'V')

# Columns required from the per-station observation CSV files.
STA_COLS = [
    'Station_Id_C', 'Datetime', 'PRE_1h',
    'WIN_D_Avg_10mi', 'WIN_S_Avg_10mi',
]


def load_config() -> dict:
    """
    Load the 2411 experiment configuration and merge local paths.

    Returns:
        dict: Merged configuration dictionary.

    Raises:
        FileNotFoundError: If the base configuration file is missing.
    """
    config_dir = pathlib.Path(__file__).parent.parent / 'config'
    base_path = config_dir / CONFIG_NAME
    if not base_path.exists():
        raise FileNotFoundError(f'Missing configuration: {base_path}')
    with open(base_path, 'r', encoding='utf-8') as file_handle:
        cfg = yaml.safe_load(file_handle)

    local_path = config_dir / LOCAL_NAME
    if local_path.exists():
        with open(local_path, 'r', encoding='utf-8') as file_handle:
            local = yaml.safe_load(file_handle)
        _deep_merge(base=cfg, override=local)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge two dictionaries; override wins.

    Args:
        base: Base dictionary, modified in place.
        override: Dictionary whose values take precedence.

    Returns:
        The merged dictionary.
    """
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def lead_hours(cfg: dict) -> np.ndarray:
    """
    Return the forecast hour of each of the 52 lead-time slots.

    Args:
        cfg: Configuration dictionary.

    Returns:
        np.ndarray: Lead hours, shape (52,).
    """
    lead = cfg['lead']
    seg_a = (
        lead['seg_a_start']
        + np.arange(lead['seg_a_count']) * lead['seg_a_step']
    )
    n_total = cfg['constants'].get('n_leads', 52)
    n_seg_b = n_total - lead['seg_a_count']
    seg_b = lead['seg_b_start'] + np.arange(n_seg_b) * lead['seg_b_step']
    return np.concatenate([seg_a, seg_b]).astype(np.float64)


def target_hours(cfg: dict) -> np.ndarray:
    """
    Return the hourly hours inside the evaluation window.

    Args:
        cfg: Configuration dictionary.

    Returns:
        np.ndarray: Hours from begin_hour + 1 to end_hour, inclusive.
    """
    begin = cfg['lead']['begin_hour']
    end = cfg['lead']['end_hour']
    return np.arange(begin + 1, end + 1, dtype=np.float64)


def _slot_slice(cfg: dict) -> typing.Tuple[int, int]:
    """
    Return the first and last lead-time slot covering the evaluation window.

    Args:
        cfg: Configuration dictionary.

    Returns:
        typing.Tuple[int, int]: First and last slot index (inclusive).
    """
    src = lead_hours(cfg)
    tgt = target_hours(cfg)
    first = int(np.searchsorted(src, tgt[0], side='right') - 1)
    last = int(np.searchsorted(src, tgt[-1], side='right') - 1)
    return max(first, 0), min(last, len(src) - 1)


def interp_weights(cfg: dict) -> typing.Tuple[np.ndarray, int, int]:
    """
    Build the linear interpolation weights from slots to target hours.

    Args:
        cfg: Configuration dictionary.

    Returns:
        typing.Tuple[np.ndarray, int, int]: Weight matrix with shape
        (n_target, n_slot_used), the first slot index and the last slot
        index (inclusive).
    """
    src = lead_hours(cfg)
    tgt = target_hours(cfg)
    first, last = _slot_slice(cfg)
    used = src[first:last + 1]
    weights = np.zeros((len(tgt), len(used)), dtype=np.float32)
    for i, hour in enumerate(tgt):
        j = int(np.searchsorted(used, hour, side='right') - 1)
        j = min(max(j, 0), len(used) - 2)
        frac = (hour - used[j]) / (used[j + 1] - used[j])
        weights[i, j] = 1.0 - frac
        weights[i, j + 1] = frac
    return weights, first, last


def time_steps(cfg: dict) -> np.ndarray:
    """
    Return the lead time length in hours for every slot.

    Args:
        cfg: Configuration dictionary.

    Returns:
        np.ndarray: Hours between consecutive slots, shape (52,).
    """
    lead = cfg['lead']
    step_a = np.full(lead['seg_a_count'], lead['seg_a_step'], dtype=float)
    n_total = cfg['constants'].get('n_leads', 52)
    n_seg_b = n_total - lead['seg_a_count']
    # The first 6-hourly slot spans the step from the last 3-hourly slot.
    step_b = np.full(n_seg_b, lead['seg_b_step'], dtype=float)
    return np.concatenate([step_a, step_b])


def station_ids(cfg: dict) -> typing.List[str]:
    """
    Return the station IDs that have both factor data and observations.

    Args:
        cfg: Configuration dictionary.

    Returns:
        typing.List[str]: Sorted station IDs.
    """
    data_dir = pathlib.Path(cfg['paths']['data_dir'])
    sta_dir = pathlib.Path(cfg['paths']['sta_dir'])
    prefix = len('data')
    have_data = {p.stem[prefix:] for p in data_dir.glob('data*.npy')}
    have_obs = {p.stem for p in sta_dir.glob('*.csv')}
    return sorted(have_data & have_obs)


def factor_columns(cfg: dict) -> typing.Tuple[typing.List[int], typing.List[str]]:
    """
    Return the factor column indices and their names.

    Args:
        cfg: Configuration dictionary.

    Returns:
        typing.Tuple[typing.List[int], typing.List[str]]: Column indices in
        data{sta}.npy and the matching names.

    Raises:
        ValueError: If the configured level list does not match the
            available columns.
    """
    levels = cfg['features']['levels']
    cols = list(COL_SURFACE)
    names = list(cfg['features']['surface'])

    if cfg['features']['precip']:
        cols.append(COL_PRECIP)
        names.append('PRE')

    for var in COL_LEVEL_ORDER:
        if len(COL_LEVEL[var]) != len(levels):
            raise ValueError(
                f'Level count mismatch for {var}: '
                f'{len(COL_LEVEL[var])} columns vs {len(levels)} levels'
            )
        for col, level in zip(COL_LEVEL[var], levels):
            cols.append(col)
            names.append(f'{var}_{level}')
    return cols, names


def load_factors(sta: str, cfg: dict) -> np.ndarray:
    """
    Load and interpolate the factors of one station.

    Precipitation is converted from metres per slot to mm/h before
    interpolation, so that the 3-hourly and 6-hourly slots are comparable.

    Args:
        sta: Station ID.
        cfg: Configuration dictionary.

    Returns:
        np.ndarray: Factors with shape (n_samples, n_hours, n_factors).
    """
    data_dir = pathlib.Path(cfg['paths']['data_dir'])
    raw = np.load(data_dir / 'data{0}.npy'.format(sta))
    cols, _ = factor_columns(cfg)
    weights, first, last = interp_weights(cfg)

    block = raw[:, first:last + 1, :][:, :, cols].astype(np.float32)
    if cfg['features']['precip']:
        steps = time_steps(cfg)[first:last + 1].astype(np.float32)
        pos = cols.index(COL_PRECIP)
        block[:, :, pos] = block[:, :, pos] * 1000.0 / steps[None, :]

    return np.einsum('nlf,hl->nhf', block, weights).astype(np.float32)


def _read_obs_frame(sta: str, cfg: dict):
    """
    Read and de-duplicate the hourly observation CSV of one station.

    The root folder and the 2024 sub-folder overlap in early January 2024;
    duplicates are removed by timestamp.

    Args:
        sta: Station ID.
        cfg: Configuration dictionary.

    Returns:
        pandas.DataFrame or None: Frame indexed by UTC time, or None when no
        observation file exists.
    """
    sta_dir = pathlib.Path(cfg['paths']['sta_dir'])
    candidates = [sta_dir / '{0}.csv'.format(sta)]
    extra = sta_dir / '2024' / '{0}.csv'.format(sta)
    if extra.exists():
        candidates.append(extra)

    frames = list()
    for path in candidates:
        if not path.exists():
            continue
        frames.append(pd.read_csv(path, usecols=STA_COLS, low_memory=False))
    if not frames:
        return None

    frame = pd.concat(frames, ignore_index=True)
    frame['Datetime'] = pd.to_datetime(frame['Datetime'], utc=True)
    frame = frame.drop_duplicates(subset=['Datetime'], keep='last')
    return frame.set_index('Datetime').sort_index()


def clean_wind(
    direction: np.ndarray, speed: np.ndarray, cfg: dict
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Apply the seven-step cleaning rule to one station's wind observations.

    Steps: special-code classification, physical range check, pair
    consistency, calm marking, temporal jump rejection.

    Args:
        direction: Raw wind direction (degrees).
        speed: Raw wind speed (m/s).
        cfg: Configuration dictionary.

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: Cleaned direction (calm
        samples keep the marker) and speed.
    """
    obs = cfg['obs']
    marker = obs['calm_marker']
    dir_val = np.asarray(direction, dtype=np.float64).copy()
    spd_val = np.asarray(speed, dtype=np.float64).copy()

    calm_raw = dir_val == marker
    dir_miss = (dir_val >= obs['miss_threshold']) & ~calm_raw
    spd_miss = spd_val >= obs['miss_threshold']

    # The calm marker is outside [0, 360]; blank it before the range check
    # and restore it after all other filters.
    dir_val[calm_raw] = np.nan
    dir_val = np.where(dir_val == 360.0, 0.0, dir_val)
    dir_val[(dir_val < 0.0) | (dir_val > 360.0)] = np.nan
    spd_val[(spd_val < 0.0) | (spd_val >= obs['max_speed'])] = np.nan

    dir_val[spd_miss] = np.nan
    spd_val[spd_miss] = np.nan
    dir_val[dir_miss] = np.nan

    jump = np.zeros_like(spd_val)
    jump[1:] = np.abs(np.diff(spd_val))
    bad = jump > obs['max_jump']
    dir_val[bad] = np.nan
    spd_val[bad] = np.nan

    calm = calm_raw & (spd_val < obs['calm_speed'])
    dir_val[calm] = marker
    return dir_val, spd_val


def sample_years(cfg: dict) -> np.ndarray:
    """
    Return the year of the forecast initial time of every sample.

    Returns:
        np.ndarray: Year per sample, shape (n_samples,).
    """
    start = pd.Timestamp(cfg['constants']['start_time'], tz='UTC')
    n_samples = int(cfg['constants']['n_samples'])
    cycle = int(cfg['constants']['cycle_hours'])
    offsets = np.arange(n_samples) * cycle
    base = start + pd.to_timedelta(offsets, unit='h')
    return base.year.to_numpy()


def year_slices(cfg: dict) -> typing.Dict[int, typing.Tuple[int, int]]:
    """
    Return the sample index range of every year (leave-one-year-out folds).

    The number of forecast cycles per year is 730 or 732 depending on leap
    years, so the ranges are derived from the initial times instead of
    assuming a fixed count.

    Returns:
        typing.Dict[int, typing.Tuple[int, int]]: Year mapped to
        (start, end) indices, end exclusive.
    """
    years = sample_years(cfg)
    out = {}
    for year in sorted(set(years.tolist())):
        idx = np.where(years == year)[0]
        out[int(year)] = (int(idx[0]), int(idx[-1]) + 1)
    return out



def sample_times(cfg: dict) -> pd.DatetimeIndex:
    """
    Return the UTC valid times of all evaluation samples.

    Returns:
        pandas.DatetimeIndex: Flattened times, length n_samples * n_hours.
    """
    start = pd.Timestamp(cfg['constants']['start_time'], tz='UTC')
    n_samples = int(cfg['constants']['n_samples'])
    cycle = int(cfg['constants']['cycle_hours'])
    hours = target_hours(cfg)
    offsets = np.arange(n_samples)[:, None] * cycle + hours[None, :]
    return start + pd.to_timedelta(offsets.ravel(), unit='h')


def load_obs(
    sta: str, cfg: dict
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load observations of one station aligned to the evaluation samples.

    Args:
        sta: Station ID.
        cfg: Configuration dictionary.

    Returns:
        typing.Tuple[np.ndarray, np.ndarray, np.ndarray]: Direction, speed
        and hourly precipitation, each with shape (n_samples, n_hours).
        All three are None when the station has no observation file.

    Raises:
        ValueError: If the time grid does not match the sample shape.
    """
    frame = _read_obs_frame(sta, cfg)
    if frame is None or frame.empty:
        return None, None, None

    dir_val, spd_val = clean_wind(
        frame['WIN_D_Avg_10mi'].values,
        frame['WIN_S_Avg_10mi'].values,
        cfg,
    )
    pre_val = frame['PRE_1h'].to_numpy(dtype=np.float64, copy=True)
    pre_val[pre_val >= cfg['obs']['miss_threshold']] = np.nan

    times = sample_times(cfg)
    shape = (int(cfg['constants']['n_samples']), len(target_hours(cfg)))
    if times.size != shape[0] * shape[1]:
        raise ValueError('Sample time grid does not match the sample shape')

    dir_out = pd.Series(dir_val, index=frame.index).reindex(times)
    spd_out = pd.Series(spd_val, index=frame.index).reindex(times)
    pre_out = pd.Series(pre_val, index=frame.index).reindex(times)
    return (
        dir_out.to_numpy().reshape(shape),
        spd_out.to_numpy().reshape(shape),
        pre_out.to_numpy().reshape(shape),
    )


def encode_sc(
    direction: np.ndarray, speed: np.ndarray, cfg: dict
) -> np.ndarray:
    """
    Encode wind direction/speed as (speed, sin, cos) targets.

    Calm samples keep the marker in the speed slot and store the observed
    speed in the sine slot, following the original implementation.

    Args:
        direction: Wind direction in degrees (calm marker allowed).
        speed: Wind speed in m/s.
        cfg: Configuration dictionary.

    Returns:
        np.ndarray: Array with trailing dimension 3.
    """
    marker = cfg['obs']['calm_marker']
    calm = direction == marker
    speed_out = np.where(calm, marker, speed)
    sin_out = np.where(calm, speed, np.sin(np.deg2rad(direction)))
    cos_out = np.where(calm, np.nan, np.cos(np.deg2rad(direction)))
    return np.stack([speed_out, sin_out, cos_out], axis=-1).astype(np.float32)


def encode_uv(
    direction: np.ndarray, speed: np.ndarray, cfg: dict
) -> np.ndarray:
    """
    Encode wind direction/speed as (u, v) components.

    The convention is u = -W*sin(theta), v = -W*cos(theta), where theta is
    the direction the wind comes from.

    Args:
        direction: Wind direction in degrees (calm marker allowed).
        speed: Wind speed in m/s.
        cfg: Configuration dictionary.

    Returns:
        np.ndarray: Array with trailing dimension 2.
    """
    marker = cfg['obs']['calm_marker']
    theta = np.deg2rad(np.where(direction == marker, np.nan, direction))
    u_val = -speed * np.sin(theta)
    v_val = -speed * np.cos(theta)
    return np.stack([u_val, v_val], axis=-1).astype(np.float32)


def build_station(sta: str, cfg: dict) -> typing.Optional[dict]:
    """
    Build the cached arrays of one station.

    Args:
        sta: Station ID.
        cfg: Configuration dictionary.

    Returns:
        typing.Optional[dict]: Dictionary with keys 'x', 'y', 'uv', 'dir',
        'spd' and 'pre', or None when observations are unavailable.
    """
    direction, speed, precip = load_obs(sta, cfg)
    if direction is None:
        return None
    return {
        'x': load_factors(sta, cfg),
        'y': encode_sc(direction, speed, cfg),
        'uv': encode_uv(direction, speed, cfg),
        'dir': direction.astype(np.float32),
        'spd': speed.astype(np.float32),
        'pre': precip.astype(np.float32),
    }


def cache_path(sta: str, cfg: dict) -> pathlib.Path:
    """
    Return the cache file path of one station.

    Args:
        sta: Station ID.
        cfg: Configuration dictionary.

    Returns:
        pathlib.Path: Cache file path.
    """
    return pathlib.Path(cfg['paths']['cache_dir']) / 'cache{0}.npz'.format(sta)


def save_cache(sta: str, cfg: dict, arrays: dict) -> pathlib.Path:
    """
    Save the cached arrays of one station.

    Args:
        sta: Station ID.
        cfg: Configuration dictionary.
        arrays: Arrays returned by build_station.

    Returns:
        pathlib.Path: Written cache file path.
    """
    path = cache_path(sta, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_cache(sta: str, cfg: dict) -> typing.Optional[dict]:
    """
    Load the cached arrays of one station.

    Args:
        sta: Station ID.
        cfg: Configuration dictionary.

    Returns:
        typing.Optional[dict]: Cached arrays, or None when missing.
    """
    path = cache_path(sta, cfg)
    if not path.exists():
        return None
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


