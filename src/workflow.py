#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data access layer for the station experiment.

Reads the per-station NWP factor arrays (data{sta}.npy) and the per-station
hourly observation CSVs (sta/{sta}.csv, plus sta/2024/{sta}.csv), cleans the
observations, interpolates the 3-hourly factors to hourly values inside the
evaluation window and encodes wind direction as (speed, sin, cos).

Founded in 2026-10-29
Modified in 2026-11-24
@author: yinlb
"""

import json
import pathlib
import typing

import numpy as np
import pandas as pd

from wind10m.cache import CACHE_FULL_KEYS, CACHE_MODES, CACHE_OBS_KEYS
from wind10m.config import deep_merge
from wind10m.config import load_config as load_workflow_config

CONFIG_NAME = 'datasets/2411.yaml'
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

# Cache descriptors already reported, so a rebuild is announced only once.
_CACHE_WARNED = set()


def load_config() -> dict:
    """
    Load generic defaults, the selected dataset, and local overrides.

    Returns:
        dict: Merged configuration dictionary.
    """
    return load_workflow_config()


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Merge configuration dictionaries through the generic implementation.

    Args:
        base: Base dictionary, modified in place.
        override: Dictionary whose values take precedence.

    Returns:
        dict: The merged dictionary.
    """
    return deep_merge(base, override)


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
    Return the evaluation grid declared by the dataset configuration.

    The grid belongs to the dataset, so another dataset simply declares its
    own ``lead.grid_start``, ``lead.grid_end`` and ``lead.grid_step``.

    Args:
        cfg: Configuration dictionary.

    Returns:
        np.ndarray: Target hours.
    """
    lead = cfg['lead']
    start = lead['grid_start']
    end = lead['grid_end']
    step = lead['grid_step']
    return np.arange(start, end + step, step, dtype=np.float64)


def grid_weights(cfg: dict) -> np.ndarray:
    """
    Build the interpolation weights for the dataset evaluation grid.

    The target hours come from ``target_hours``; weights are computed against
    the full source slot grid so that any declared evaluation grid shares one
    implementation.

    Args:
        cfg: Configuration dictionary.

    Returns:
        np.ndarray: Weight matrix with shape (n_target, 52), float32.
    """
    src = lead_hours(cfg)
    tgt = target_hours(cfg)
    weights = np.zeros((tgt.size, src.size), dtype=np.float32)
    for index, hour in enumerate(tgt):
        left = int(np.searchsorted(src, hour, side='right') - 1)
        left = min(max(left, 0), src.size - 2)
        frac = (hour - src[left]) / (src[left + 1] - src[left])
        weights[index, left] = 1.0 - frac
        weights[index, left + 1] = frac
    return weights


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


def available_ids(cfg: dict) -> typing.List[str]:
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


def station_ids(cfg: dict) -> typing.List[str]:
    """
    Return the station IDs used by training and verification.

    When ``paths.station_list`` is set, only IDs in that required QC list are
    kept. Leave it unset only for an intentional unfiltered experiment.

    Args:
        cfg: Configuration dictionary.

    Returns:
        typing.List[str]: Sorted station IDs.

    Raises:
        FileNotFoundError: If the configured QC station list is missing.
    """
    station_all = available_ids(cfg)
    list_path = cfg['paths'].get('station_list')
    if list_path:
        path = pathlib.Path(list_path)
        if not path.exists():
            raise FileNotFoundError(
                'QC station list not found: {0}. Run qc_stations.py '
                'first or unset paths.station_list intentionally.'.format(path)
            )
        with open(path, 'r', encoding='utf-8') as file_handle:
            keep = {
                line.strip() for line in file_handle if line.strip()
            }
        station_all = [sta for sta in station_all if sta in keep]
    return station_all


def factor_columns(
    cfg: dict
) -> typing.Tuple[typing.List[int], typing.List[str]]:
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
    weights = grid_weights(cfg)

    block = raw[:, :, cols].astype(np.float32)
    if cfg['features']['precip']:
        steps = time_steps(cfg).astype(np.float32)
        pos = cols.index(COL_PRECIP)
        # Precipitation is non-negative; small negative values come from the
        # source field and are clipped to zero before the unit conversion.
        precip = np.maximum(block[:, :, pos], 0.0)
        block[:, :, pos] = precip * 1000.0 / steps[None, :]

    return np.einsum('nlf,hl->nhf', block, weights).astype(np.float32)


def _read_obs_frame(sta: str, cfg: dict) -> pd.DataFrame:
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
    dir_val = np.asarray(direction, dtype=np.float32).copy()
    spd_val = np.asarray(speed, dtype=np.float32).copy()

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
    pre_val = frame['PRE_1h'].to_numpy(dtype=np.float32, copy=True)
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


def cache_mode(cfg: dict) -> str:
    """
    Return the validated persistent-cache mode.

    Args:
        cfg: Configuration dictionary.

    Returns:
        str: 'none', 'obs', or 'full'.

    Raises:
        ValueError: If cache.mode is not a supported value.
    """
    mode = str(cfg.get('cache', {}).get('mode', 'none')).lower()
    if mode not in CACHE_MODES:
        raise ValueError(
            "cache.mode must be one of 'none', 'obs', or 'full'; "
            'got {0!r}'.format(mode)
        )
    return mode


def build_observations(sta: str, cfg: dict) -> typing.Optional[dict]:
    """
    Build the aligned observation arrays of one station.

    Args:
        sta: Station ID.
        cfg: Configuration dictionary.

    Returns:
        typing.Optional[dict]: Float32 direction, speed, and precipitation
        arrays, or None when observations are unavailable.
    """
    direction, speed, precip = load_obs(sta, cfg)
    if direction is None:
        return None
    return {
        'dir': direction.astype(np.float32, copy=False),
        'spd': speed.astype(np.float32, copy=False),
        'pre': precip.astype(np.float32, copy=False),
    }


def build_station(sta: str, cfg: dict) -> typing.Optional[dict]:
    """
    Build all in-memory arrays of one station.

    Args:
        sta: Station ID.
        cfg: Configuration dictionary.

    Returns:
        typing.Optional[dict]: Dictionary with keys 'x', 'y', 'uv', 'dir',
        'spd' and 'pre', or None when observations are unavailable.
    """
    arrays = build_observations(sta, cfg)
    if arrays is None:
        return None
    arrays['x'] = load_factors(sta, cfg)
    arrays['y'] = encode_sc(arrays['dir'], arrays['spd'], cfg)
    arrays['uv'] = encode_uv(arrays['dir'], arrays['spd'], cfg)
    return arrays


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


def pred_root(cfg: dict) -> pathlib.Path:
    """
    Return the prediction directory.

    Args:
        cfg: Configuration dictionary.

    Returns:
        pathlib.Path: Prediction directory.
    """
    return pathlib.Path(cfg['paths']['pred_dir'])


def result_root(cfg: dict) -> pathlib.Path:
    """
    Return the result directory.

    Args:
        cfg: Configuration dictionary.

    Returns:
        pathlib.Path: Result directory.
    """
    return pathlib.Path(cfg['paths']['result_dir'])


def cache_descriptor(cfg: dict) -> str:
    """
    Return the JSON descriptor stored with every cache file.

    The descriptor records the cache mode, evaluation hours, feature count and
    grid, so a cache built for another configuration is never reused silently.

    Args:
        cfg: Configuration dictionary.

    Returns:
        str: Stable JSON descriptor.
    """
    lead = cfg['lead']
    return json.dumps(
        {
            'mode': cache_mode(cfg),
            'n_hour': int(target_hours(cfg).size),
            'n_feature': len(factor_columns(cfg)[0]),
            'grid': [
                float(lead['grid_start']),
                float(lead['grid_end']),
                float(lead['grid_step']),
            ],
        },
        sort_keys=True,
    )


def _restore_arrays(
    sta: str, stored: dict, cfg: dict
) -> typing.Optional[dict]:
    """Restore a complete station dictionary from any cache format."""
    if 'dir' not in stored or 'spd' not in stored:
        return None
    n_hour = int(target_hours(cfg).size)
    if stored['dir'].shape[1] != n_hour:
        return None
    if 'meta' in stored and str(stored['meta']) != cache_descriptor(cfg):
        return None

    arrays = {
        key: stored[key] for key in CACHE_OBS_KEYS if key in stored
    }
    if 'x' in stored:
        if stored['x'].shape[1] != n_hour:
            return None
        arrays['x'] = stored['x']
    else:
        arrays['x'] = load_factors(sta, cfg)
    if 'y' in stored:
        arrays['y'] = stored['y']
    else:
        arrays['y'] = encode_sc(arrays['dir'], arrays['spd'], cfg)
    if 'uv' in stored:
        arrays['uv'] = stored['uv']
    else:
        arrays['uv'] = encode_uv(arrays['dir'], arrays['spd'], cfg)
    return arrays


def _warn_cache(
    path: pathlib.Path, stored: dict, cfg: dict
) -> None:
    """Report an existing cache that cannot serve the active configuration."""
    hours = int(target_hours(cfg).size)
    if hours in _CACHE_WARNED:
        return
    _CACHE_WARNED.add(hours)
    found = stored['dir'].shape[1] if 'dir' in stored else None
    print(
        'Cache not reusable ({0} holds {1} hours, {2} required); rebuilding '
        'in memory. Set cache.mode and run build_cache.py to refresh '
        '{3}'.format(path.name, found, hours, path.parent),
        flush=True,
    )


def load_arrays(sta: str, cfg: dict) -> typing.Optional[dict]:
    """
    Return complete station arrays for the active target grid.

    Reuse a compatible full, observation-only, or legacy cache. Missing
    factors and encoded targets are rebuilt in memory. If no compatible cache
    exists, build all arrays from the raw inputs without writing a cache.

    Args:
        sta: Station ID.
        cfg: Configuration dictionary.

    Returns:
        typing.Optional[dict]: Arrays with keys 'x', 'y', 'uv', 'dir', 'spd'
        and 'pre', or None when observations are unavailable.
    """
    path = cache_path(sta, cfg)
    if path.exists():
        with np.load(path) as data:
            stored = {key: data[key] for key in data.files}
        arrays = _restore_arrays(sta, stored, cfg)
        if arrays is not None:
            return arrays
        _warn_cache(path, stored, cfg)
    return build_station(sta, cfg)


def save_cache(sta: str, cfg: dict, arrays: dict) -> pathlib.Path:
    """
    Save station arrays according to cache.mode.

    Full mode stores x/dir/spd/pre, while observation mode stores only
    dir/spd/pre. Encoded y/uv targets are derived from dir/spd on loading.
    Legacy caches containing all arrays remain readable.

    Args:
        sta: Station ID.
        cfg: Configuration dictionary.
        arrays: Arrays returned by build_station or build_observations.

    Returns:
        pathlib.Path: Written cache file path.

    Raises:
        ValueError: If caching is disabled or a required array is missing.
    """
    mode = cache_mode(cfg)
    if mode == 'none':
        raise ValueError(
            "Cache building is disabled; set cache.mode to 'obs' or 'full'"
        )
    keys = CACHE_FULL_KEYS if mode == 'full' else CACHE_OBS_KEYS
    missing = [key for key in keys if key not in arrays]
    if missing:
        raise ValueError('Missing cache arrays: {0}'.format(missing))
    path = cache_path(sta, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: arrays[key] for key in keys}
    payload['meta'] = np.array(cache_descriptor(cfg))
    # Uncompressed npz: float32 factors compress poorly, and decompression
    # dominates training I/O when station files are read repeatedly.
    np.savez(path, **payload)
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


