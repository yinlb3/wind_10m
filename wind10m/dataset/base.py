#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface implemented by replaceable wind datasets.

Founded in 2026-11-23
Modified in 2026-11-24
@author: yinlb
"""

import abc
import pathlib
import typing


class DatasetAdapter(abc.ABC):
    """Provide the data operations required by the generic workflow."""

    def __init__(self, cfg: dict) -> None:
        """Store the merged dataset configuration."""
        self.cfg = cfg

    @property
    def n_samples(self) -> int:
        """Return the number of forecast samples per station."""
        return int(self.cfg['constants']['n_samples'])

    @abc.abstractmethod
    def available_ids(self) -> typing.List[str]:
        """Return stations having both model factors and observations."""

    @abc.abstractmethod
    def station_ids(self) -> typing.List[str]:
        """Return available stations after the configured QC filtering."""

    @abc.abstractmethod
    def target_hours(self):
        """Return the target forecast-hour grid."""

    @abc.abstractmethod
    def feature_count(self) -> int:
        """Return the number of model input features."""

    @abc.abstractmethod
    def build_observations(self, station: str) -> typing.Optional[dict]:
        """Build aligned observation arrays for one station."""

    @abc.abstractmethod
    def build_station(self, station: str) -> typing.Optional[dict]:
        """Build all in-memory arrays for one station."""

    @abc.abstractmethod
    def load_arrays(self, station: str) -> typing.Optional[dict]:
        """Load or build all arrays for one station."""

    @abc.abstractmethod
    def cache_path(self, station: str) -> pathlib.Path:
        """Return the persistent-cache path for one station."""
