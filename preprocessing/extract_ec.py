#!user/bin.python3
# -*- coding: utf-8 -*-

"""
ECMWF-IFS data extraction module

Extract meteorological elements from EC4AI binary files to specified stations
and output CSV format data.

Founded in 2023-07-15
Modified in 2026-07-21
@author: yinlb
"""

import os
import sys
from pathlib import Path

import arrow
import numpy as np
import pandas as pd
from meteva import base as meb

from src import utils


# EC element list to extract (30 in total)
# Surface elements: 2T, MSL, 2D, 10U, 10V
# Level elements (1000/950/925/900/850 hPa): T, GH, R, U, V
ELEMENTS = (
    # Surface elements
    '2T', 'MSL', '2D', '10U', '10V',
    # 1000 hPa
    'T_1000', 'GH_1000', 'R_1000', 'U_1000', 'V_1000',
    # 950 hPa
    'T_950', 'GH_950', 'R_950', 'U_950', 'V_950',
    # 925 hPa
    'T_925', 'GH_925', 'R_925', 'U_925', 'V_925',
    # 900 hPa
    'T_900', 'GH_900', 'R_900', 'U_900', 'V_900',
    # 850 hPa
    'T_850', 'GH_850', 'R_850', 'U_850', 'V_850'
)

# Surface elements use the high-resolution 0.125-degree grid.
N_SURFACE_ELEMENTS = 5


def main(
    input_path: str,
    output_path: str,
    station_info_path: str,
) -> None:
    """
    Extract data from EC4AI to stations.

    Iterate over the specified time range of EC forecast data,
    extract 30 elements to 97 stations, and output CSV files.

    Args:
        input_path: Root directory of EC4AI binary files.
        output_path: Output CSV file directory.
        station_info_path: Station information CSV file path
            (contains station ID, longitude, and latitude columns).

    Returns:
        None
    """
    # ---- 1. Read station information and define grids ----
    station_df = pd.read_csv(
        station_info_path,
        encoding='gb2312',
        low_memory=False,
    )
    station_metadata = meb.sta_data(
        station_df.loc[:, ('台站号', '经度', '纬度')],
        columns=('id', 'lon', 'lat'),
    )
    # High-resolution grid (0.125 deg) for surface elements and standard
    # grid (0.25 deg) for level elements.
    grid_high_res = meb.grid((70, 140, 0.125), (0, 60, 0.125))
    grid_standard_res = meb.grid((70, 140, 0.25), (0, 60, 0.25))

    input_dir = Path(input_path)
    output_dir = Path(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # ---- 2. Iterate over forecast initial times and extract data ----
    time_begin = arrow.get('2017')
    time_end = arrow.get('2023')
    forecast_ref_time = time_begin
    while forecast_ref_time < time_end:
        year_month = forecast_ref_time.format('YYYYMM')
        for lead_time in range(12, 37, 3):
            init_time = forecast_ref_time.format('MMDDHH')
            valid_time = forecast_ref_time.shift(
                hours=lead_time
            ).format('MMDDHH')
            ec_filename_base = 'C1D{:s}00{:s}001'.format(init_time, valid_time)
            station_combined = None
            for elem_idx, element in enumerate(ELEMENTS):
                filename = '{:s}_{:s}.AI.bin'.format(ec_filename_base, element)
                file_path = (
                    input_dir / year_month / ec_filename_base / filename
                )
                if not file_path.exists():
                    station_combined = None
                    break
                raw_data = np.fromfile(str(file_path), dtype=np.float32)
                if elem_idx < N_SURFACE_ELEMENTS:
                    grid_data = meb.grid_data(grid_high_res, raw_data)
                else:
                    grid_data = meb.grid_data(grid_standard_res, raw_data)
                station_single = meb.interp_gs_nearest(
                    grid_data, station_metadata
                )
                meb.set_stadata_names(station_single, element)
                meb.set_stadata_coords(
                    station_single,
                    level=0,
                    time=forecast_ref_time.datetime,
                    dtime=lead_time,
                )
                station_combined = meb.combine_on_id(
                    station_combined, station_single
                )
            if station_combined is not None:
                output_filename = '{:s}.{:03d}.csv'.format(
                    forecast_ref_time.format('YYYYMMDDHH'), lead_time
                )
                station_combined.to_csv(
                    output_dir / output_filename, index=False
                )
        forecast_ref_time = forecast_ref_time.shift(hours=12)


if __name__ == '__main__':
    print('Program extract_ec.py started')
    total_start = arrow.now()

    if len(sys.argv) == 1:
        cfg = utils.load_config()
        paths = cfg['paths']
        main(
            input_path=paths['ec4ai'],
            output_path=paths['ec_station'],
            station_info_path=paths['station_csv'],
        )
    elif len(sys.argv) == 4:
        main(
            input_path=sys.argv[1],
            output_path=sys.argv[2],
            station_info_path=sys.argv[3],
        )

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(
        'Program extract_ec.py finished, total time '
        f'{utils.format_time(total_elapsed)}'
    )
