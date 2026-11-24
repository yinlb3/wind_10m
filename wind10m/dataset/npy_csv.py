#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapter for per-station NPY factors and hourly CSV observations.

Founded in 2026-11-23
Modified in 2026-11-24
@author: yinlb
"""

import pathlib
import typing

from src import workflow
from wind10m.dataset.base import DatasetAdapter


class NpyCsvAdapter(DatasetAdapter):
    """Delegate reference operations without changing numerical behavior."""

    def available_ids(self) -> typing.List[str]:
        """Return the raw-data station intersection."""
        return workflow.available_ids(self.cfg)

    def station_ids(self) -> typing.List[str]:
        """Return stations retained by the configured QC list."""
        return workflow.station_ids(self.cfg)

    def target_hours(self):
        """Return the configured target forecast-hour grid."""
        return workflow.target_hours(self.cfg)

    def feature_count(self) -> int:
        """Return the configured model feature count."""
        return len(workflow.factor_columns(self.cfg)[0])

    def build_observations(self, station: str) -> typing.Optional[dict]:
        """Build aligned observations for one station."""
        return workflow.build_observations(station, self.cfg)

    def build_station(self, station: str) -> typing.Optional[dict]:
        """Build all arrays for one station."""
        return workflow.build_station(station, self.cfg)

    def load_arrays(self, station: str) -> typing.Optional[dict]:
        """Load compatible cache arrays or rebuild them in memory."""
        return workflow.load_arrays(station, self.cfg)

    def cache_path(self, station: str) -> pathlib.Path:
        """Return the cache path for one station."""
        return workflow.cache_path(station, self.cfg)
