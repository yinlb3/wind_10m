#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Failure-injection tests for the resumable CMADaaS downloader.

Founded in 2026-11-23
Modified in 2026-11-24
@author: yinlb
"""

import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import arrow
import pandas as pd

import download_obs as download


class DownloadTest(unittest.TestCase):
    """Exercise recovery without making real network requests."""

    def setUp(self) -> None:
        """Create a temporary two-station input file and output directory."""
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        self.output = self.root / 'out'
        self.station_info = self.root / 'stations.csv'
        pd.DataFrame({'台站号': ['00001', '00002']}).to_csv(
            self.station_info, index=False, encoding='gb2312'
        )

    def tearDown(self) -> None:
        """Remove only this test's temporary directory."""
        self.temp.cleanup()

    @staticmethod
    def response(period: str, station_ids) -> pd.DataFrame:
        """Build one small station-batch response."""
        ids = list(station_ids)
        return pd.DataFrame({
            'Station_Id_C': ids,
            'Datetime': [period] * len(ids),
            'WIN_S_Avg_10mi': [1.0] * len(ids),
        })

    def run_main(self, fake_fetch, batch_size=2):
        """Run the real main workflow with one mocked API function."""
        output = io.StringIO()
        with mock.patch.object(
            download.rc,
            'cmadaas_obs_by_time_range_and_id',
            side_effect=fake_fetch,
        ), mock.patch.object(download, 'tqdm', None), mock.patch.object(
            download, '_progress', return_value=None
        ), contextlib.redirect_stdout(output):
            incomplete = download.main(
                output_path=str(self.output),
                station_info_path=str(self.station_info),
                time_start=arrow.get('20210101', 'YYYYMMDD'),
                time_end=arrow.get('20220101', 'YYYYMMDD'),
                batch_size=batch_size,
                max_attempts=2,
                retry_wait=0,
            )
        return incomplete, output.getvalue()

    def test_empty_response_resume_preserves_and_deduplicates(self) -> None:
        """An empty slice stays unresolved and resumes without repeats."""
        first_calls = []

        def first_fetch(time_range, sta_ids, **kwargs):
            first_calls.append((time_range, sta_ids))
            if time_range.startswith('[20210101'):
                return None
            return self.response('20210701', sta_ids.split(','))

        incomplete, _ = self.run_main(first_fetch)
        self.assertEqual(incomplete, ['00001', '00002'])
        self.assertEqual(len(first_calls), 2)
        self.assertFalse((self.output / '00001.csv').exists())
        temp_path = self.output / '00002.csv.tmp'
        self.assertTrue(temp_path.exists())
        with open(temp_path, 'ab') as handle:
            handle.write(b'uncheckpointed')

        second_calls = []

        def second_fetch(time_range, sta_ids, **kwargs):
            second_calls.append((time_range, sta_ids))
            return self.response('20210101', sta_ids.split(','))

        incomplete, _ = self.run_main(second_fetch)
        self.assertEqual(incomplete, [])
        self.assertEqual(len(second_calls), 1)
        self.assertTrue(second_calls[0][0].startswith('[20210101'))
        for sid in ('00001', '00002'):
            final_path = self.output / f'{sid}.csv'
            self.assertTrue(final_path.exists())
            self.assertEqual(len(pd.read_csv(final_path)), 2)
            self.assertFalse((self.output / f'{sid}.csv.tmp').exists())
        with open(self.output / download.STATE_NAME, encoding='utf-8') as file:
            state = json.load(file)
        self.assertEqual(state['failures'], {})

    def test_station_without_any_rows_gets_header_only_csv(self) -> None:
        """Absence from valid responses produces an empty station CSV."""

        def fetch(time_range, sta_ids, **kwargs):
            del time_range
            return self.response('20210101', ['00001'])

        incomplete, _ = self.run_main(fetch)
        self.assertEqual(incomplete, [])
        self.assertEqual(len(pd.read_csv(self.output / '00001.csv')), 2)
        empty = pd.read_csv(self.output / '00002.csv')
        self.assertEqual(len(empty), 0)
        self.assertEqual(tuple(empty.columns), download.CSV_COLUMNS)

    def test_error_does_not_block_later_station_batch(self) -> None:
        """A failed station batch does not erase or stop successful batches."""
        calls = []

        def fetch(time_range, sta_ids, **kwargs):
            calls.append((time_range, sta_ids))
            if sta_ids == '00001':
                raise RuntimeError('synthetic request error')
            return self.response('20210701', ['00002'])

        incomplete, output = self.run_main(fetch, batch_size=1)
        self.assertEqual(incomplete, ['00001'])
        self.assertTrue((self.output / '00002.csv').exists())
        self.assertEqual(len(pd.read_csv(self.output / '00002.csv')), 2)
        self.assertTrue((self.output / download.STATE_NAME).exists())
        self.assertTrue((self.output / download.LOG_NAME).exists())
        self.assertIn('incomplete=1', output)
        self.assertTrue(any(item[1] == '00002' for item in calls))


if __name__ == '__main__':
    unittest.main()
