"""Adapter for per-station NPY factors and hourly CSV observations."""

import pathlib
import typing

from src import data_2411
from wind10m.dataset.base import DatasetAdapter


class NpyCsvAdapter(DatasetAdapter):
    """Delegate reference operations without changing numerical behavior."""

    def available_station_ids(self) -> typing.List[str]:
        """Return the raw-data station intersection."""
        return data_2411.available_station_ids(self.cfg)

    def station_ids(self) -> typing.List[str]:
        """Return stations retained by the configured QC list."""
        return data_2411.station_ids(self.cfg)

    def target_hours(self):
        """Return the configured target forecast-hour grid."""
        return data_2411.target_hours(self.cfg)

    def feature_count(self) -> int:
        """Return the configured model feature count."""
        return len(data_2411.factor_columns(self.cfg)[0])

    def build_observations(self, station: str) -> typing.Optional[dict]:
        """Build aligned observations for one station."""
        return data_2411.build_observations(station, self.cfg)

    def build_station(self, station: str) -> typing.Optional[dict]:
        """Build all arrays for one station."""
        return data_2411.build_station(station, self.cfg)

    def load_arrays(self, station: str) -> typing.Optional[dict]:
        """Load compatible cache arrays or rebuild them in memory."""
        return data_2411.load_arrays(station, self.cfg)

    def cache_path(self, station: str) -> pathlib.Path:
        """Return the cache path for one station."""
        return data_2411.cache_path(station, self.cfg)
