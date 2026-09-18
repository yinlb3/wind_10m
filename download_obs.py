#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMADaaS station observation download module

Download hourly surface observations for national surface stations from
CMADaaS and write one CSV per station, producing the raw per-station CSV
files consumed by extract_obs.py.

Founded in 2026-07-29
Modified in 2026-09-17
@author: yinlb, deepseek-flash
"""

import os
import pathlib
import sys
import typing

import arrow
from meteva import base as meb
from nmc_met_io import retrieve_cmadaas as rc

from src import utils

# CMADaaS element list to download (8 columns). These are exactly the raw
# columns that extract_obs.py maps to the target observation columns
# (see RAW_COLUMNS/COLUMN_MAP in extract_obs.py).
ELEMENTS = (
    'Station_Id_C,Datetime,'
    'WIN_D_Avg_2mi,WIN_S_Avg_2mi,'
    'WIN_D_Avg_10mi,WIN_S_Avg_10mi,'
    'WIN_D_INST_Max,WIN_S_Inst_Max'
)

# CMADaaS data code for national surface station hourly observations.
DATA_CODE = 'SURF_CHN_MUL_HOR'


def main(
    output_path: str,
    years: typing.List[int],
    time_start: typing.Optional[arrow.Arrow] = None,
    time_end: typing.Optional[arrow.Arrow] = None,
) -> None:
    """
    Download per-station hourly observations from CMADaaS.

    Read the station IDs from the meteva built-in national station list
    (meb.station_国家站), download the wind elements
    for each station over the given time range, and write one
    {station_id}.csv per station. Existing station files are skipped so
    interrupted runs can be resumed.

    Args:
        output_path: Output directory for per-station CSV files.
        years: Years covered by this run; the default start time is the
            first day of years[0].
        time_start: Start of the download time range, inclusive.
            Defaults to the start of the configured year range.
        time_end: End of the download time range, exclusive.
            Defaults to the current time.
    """
    # ---- 1. Resolve the default time range ----
    if time_start is None:
        time_start = arrow.get(str(years[0]))
    if time_end is None:
        time_end = arrow.now()
    start_time = time_start.format('YYYYMMDDHHmmss')
    end_time = time_end.format('YYYYMMDDHHmmss')
    print(f'Download time range: [{start_time}, {end_time})')

    # ---- 2. Read the meteva built-in national station ID list ----
    station_df = meb.read_station(meb.station_国家站)
    sid_list = station_df['id'].astype(str).tolist()
    print(f'Loaded {len(sid_list)} stations from meteva station list')

    # ---- 3. Download one CSV per station, skipping existing files ----
    output_dir = pathlib.Path(output_path)
    os.makedirs(output_dir, exist_ok=True)
    for sid in sid_list:
        csv_path = output_dir / f'{sid}.csv'
        if csv_path.exists():
            print(f'Skip {sid}: file already exists')
            continue
        sta = rc.cmadaas_obs_by_time_range_and_id(
            time_range=f'[{start_time},{end_time})',
            data_code=DATA_CODE,
            trans_type=False,
            elements=ELEMENTS,
            sta_ids=sid,
        )
        if sta is None:
            print(f'No data returned for station {sid}')
            continue
        sta = sta.sort_values(by=['Datetime'])
        sta.to_csv(path_or_buf=csv_path, index=False)
        print(f'Saved {csv_path}')


if __name__ == '__main__':
    print('Program download_obs.py started')
    total_start = arrow.now()

    if len(sys.argv) == 1:
        cfg = utils.load_config()
        main(
            output_path=cfg['paths']['sta_raw'],
            years=cfg['constants']['years'],
        )
    elif len(sys.argv) == 4:
        cfg = utils.load_config()
        main(
            output_path=sys.argv[1],
            years=cfg['constants']['years'],
            time_start=arrow.get(sys.argv[2], 'YYYYMMDD'),
            time_end=arrow.get(sys.argv[3], 'YYYYMMDD'),
        )

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(
        'Program download_obs.py finished, total time '
        f'{utils.format_time(total_elapsed)}'
    )
