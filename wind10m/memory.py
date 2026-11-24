#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared runtime-memory helpers for the workflow entry points.

Founded in 2026-11-23
Modified in 2026-11-24
@author: yinlb
"""

BYTES_PER_GB = 1024.0 ** 3
DEFAULT_BUDGET_RATIO = 0.25
MINIMUM_BLOCK = 32
# Observation, mask and temporary arrays held per station block, expressed in
# units of one (samples x hours x stations) float32 array, excluding models.
FIXED_BLOCK_ARRAYS = 6.0


def available_gb() -> float:
    """
    Return the currently available RAM in GB.

    Returns:
        float: Available memory, or -1.0 when psutil is unavailable.
    """
    try:
        import psutil
    except ImportError:
        return -1.0
    return psutil.virtual_memory().available / BYTES_PER_GB


def station_block_size(
    n_station: int,
    n_hour: int,
    n_model: int,
    budget_ratio: float = DEFAULT_BUDGET_RATIO,
) -> int:
    """
    Choose a station block that fits the currently available memory.

    A block holds the two observation arrays, one prediction copy per model,
    the common-sample masks and a few float32 temporaries of the same shape.

    Args:
        n_station: Total number of stations.
        n_hour: Target hours per sample.
        n_model: Number of models evaluated together.
        budget_ratio: Share of the available RAM the block may use.

    Returns:
        int: Stations per block, between ``MINIMUM_BLOCK`` and ``n_station``.
        Unknown memory keeps the whole dataset in one block, which matches
        the previous behaviour for datasets that already fit in memory.
    """
    if n_station <= MINIMUM_BLOCK:
        return max(1, n_station)
    free_gb = available_gb()
    if free_gb <= 0:
        return n_station
    arrays = FIXED_BLOCK_ARRAYS + float(n_model)
    per_station = max(1.0, float(n_hour) * 4.0 * arrays)
    budget = free_gb * BYTES_PER_GB * float(budget_ratio)
    size = int(budget / per_station)
    return max(MINIMUM_BLOCK, min(n_station, size))
