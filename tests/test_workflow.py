"""Small real-data-path checks using one temporary station."""

import contextlib
import copy
import io
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import yaml

import build_cache_2411 as builder
import cache_size
import qc_stations_2411 as qc
from src import data_2411 as data
from wind10m.cache import estimate_sizes
from wind10m.config import load_config
from wind10m.dataset import NpyCsvAdapter, create_adapter


class WorkflowTest(unittest.TestCase):
    """Exercise adapters, caches, QC, and configuration without large data."""

    def setUp(self) -> None:
        """Create one readable station with 24 hourly observations."""
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        self.cfg = copy.deepcopy(load_config())
        self.cfg['constants']['n_samples'] = 1
        self.cfg['lead']['mode'] = 24
        self.cfg['paths'] = {
            'data_dir': str(self.root / 'raw'),
            'sta_dir': str(self.root / 'sta'),
            'cache_dir': str(self.root / 'cache'),
            'result_dir': str(self.root / 'results'),
            'station_list': str(self.root / 'stations_ok.txt'),
        }
        pathlib.Path(self.cfg['paths']['data_dir']).mkdir()
        pathlib.Path(self.cfg['paths']['sta_dir']).mkdir()
        raw = np.ones((1, 52, 31), dtype=np.float32)
        np.save(
            pathlib.Path(self.cfg['paths']['data_dir']) / 'data00001.npy',
            raw,
        )
        times = pd.date_range('2019-01-01 13:00', periods=24, freq='h')
        frame = pd.DataFrame({
            'Station_Id_C': '00001',
            'Datetime': times.strftime('%Y-%m-%d %H:%M:%S'),
            'PRE_1h': 0.0,
            'WIN_D_Avg_10mi': 90.0,
            'WIN_S_Avg_10mi': 3.0,
        })
        frame.to_csv(
            pathlib.Path(self.cfg['paths']['sta_dir']) / '00001.csv',
            index=False,
        )

    def tearDown(self) -> None:
        """Remove only this test's temporary directory."""
        self.temp.cleanup()

    def assert_arrays_equal(self, actual: dict, expected: dict) -> None:
        """Compare all arrays, including NaN positions, exactly."""
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_adapter_matches_reference(self) -> None:
        """The adapter must delegate without changing array values."""
        adapter = create_adapter(self.cfg)
        self.assertIsInstance(adapter, NpyCsvAdapter)
        self.assertEqual(adapter.available_station_ids(), ['00001'])
        self.assertEqual(adapter.n_samples, 1)
        self.assertEqual(adapter.feature_count(), 26)
        expected = data.build_station('00001', self.cfg)
        self.assert_arrays_equal(adapter.build_station('00001'), expected)
        self.assert_arrays_equal(adapter.load_arrays('00001'), expected)
        self.assertFalse(adapter.cache_path('00001').exists())

    def test_cache_modes_and_legacy(self) -> None:
        """Observation and full caches must round-trip exactly."""
        expected = data.build_station('00001', self.cfg)
        with mock.patch.object(data, 'load_config', return_value=self.cfg):
            self.cfg['cache']['mode'] = 'obs'
            with contextlib.redirect_stdout(io.StringIO()):
                builder.main()
            with np.load(data.cache_path('00001', self.cfg)) as stored:
                self.assertEqual(set(stored.files), set(data.CACHE_OBS_KEYS))
                payload = sum(stored[key].nbytes for key in stored.files)
            self.assertEqual(
                payload, estimate_sizes(1, 1, 24, 26)['obs']
            )
            self.assert_arrays_equal(
                data.load_arrays('00001', self.cfg), expected
            )

            self.cfg['cache']['mode'] = 'full'
            with contextlib.redirect_stdout(io.StringIO()):
                builder.main()
            with np.load(data.cache_path('00001', self.cfg)) as stored:
                self.assertEqual(set(stored.files), set(data.CACHE_FULL_KEYS))
                payload = sum(stored[key].nbytes for key in stored.files)
            self.assertEqual(
                payload, estimate_sizes(1, 1, 24, 26)['full']
            )
            self.assert_arrays_equal(
                data.load_arrays('00001', self.cfg), expected
            )

        self.cfg['cache']['mode'] = 'none'
        self.cfg['paths']['cache_dir'] = str(self.root / 'legacy')
        path = data.cache_path('00001', self.cfg)
        path.parent.mkdir(parents=True)
        np.savez(path, **expected)
        self.assert_arrays_equal(data.load_arrays('00001', self.cfg), expected)

    def test_qc_without_cache(self) -> None:
        """QC reads raw observations and creates no cache directory."""
        with mock.patch.object(data, 'load_config', return_value=self.cfg):
            with contextlib.redirect_stdout(io.StringIO()):
                qc.main()
        station_list = pathlib.Path(self.cfg['paths']['station_list'])
        self.assertEqual(station_list.read_text().strip(), '00001')
        self.assertFalse(pathlib.Path(self.cfg['paths']['cache_dir']).exists())
        self.assertEqual(cache_size.infer_station_count(self.cfg)[0], 1)

    def test_layered_config_and_unknown_adapter(self) -> None:
        """Dataset and local YAML values override generic defaults in order."""
        dataset_cfg = copy.deepcopy(self.cfg)
        dataset_cfg['constants']['n_samples'] = 2
        dataset_cfg['cache']['mode'] = 'obs'
        dataset_path = self.root / 'dataset.yaml'
        local_path = self.root / 'dataset.local.yaml'
        with open(dataset_path, 'w', encoding='utf-8') as handle:
            yaml.safe_dump(dataset_cfg, handle, sort_keys=False)
        with open(local_path, 'w', encoding='utf-8') as handle:
            yaml.safe_dump(
                {'constants': {'n_samples': 3}, 'cache': {'mode': 'full'}},
                handle,
            )
        loaded = load_config(dataset_path, local_path)
        self.assertEqual(loaded['constants']['n_samples'], 3)
        self.assertEqual(loaded['cache']['mode'], 'full')
        loaded['dataset']['type'] = 'missing'
        with self.assertRaises(ValueError):
            create_adapter(loaded)


if __name__ == '__main__':
    unittest.main()
