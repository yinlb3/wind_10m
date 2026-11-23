"""Cache format constants and dataset-independent size estimates."""

import numpy as np

CACHE_MODES = ('none', 'obs', 'full')
CACHE_OBS_KEYS = ('dir', 'spd', 'pre')
CACHE_FULL_KEYS = ('x',) + CACHE_OBS_KEYS
DTYPE = np.dtype(np.float32)


def estimate_sizes(
    stations: int, samples: int, hours: int, features: int
) -> dict:
    """
    Estimate uncompressed cache payload sizes in bytes.

    Args:
        stations: Number of stations.
        samples: Forecast samples per station.
        hours: Target hours per sample.
        features: Number of model features.

    Returns:
        dict: Total bytes for none, obs, and full modes.
    """
    values_per_station = samples * hours
    obs_per_station = (
        values_per_station * len(CACHE_OBS_KEYS) * DTYPE.itemsize
    )
    full_per_station = obs_per_station + (
        values_per_station * features * DTYPE.itemsize
    )
    return {
        'none': 0,
        'obs': stations * obs_per_station,
        'full': stations * full_per_station,
    }


def format_size(size: int) -> str:
    """
    Format a byte count with binary units.

    Args:
        size: Number of bytes.

    Returns:
        str: Human-readable size.
    """
    value = float(size)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024.0 or unit == 'TiB':
            if unit == 'B':
                return '{0:d} B'.format(int(size))
            return '{0:.2f} {1}'.format(value, unit)
        value /= 1024.0
    return '{0:.2f} TiB'.format(value)
