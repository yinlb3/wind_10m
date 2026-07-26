#!user/bin.python3
# -*- coding: utf-8 -*-

"""
Station observation extraction module

Convert new raw per-station observation CSV files (long-table format, one
file per station) directly into the observation numpy arrays (ob{year}.npy)
consumed by the model training scripts, replacing the per-hour CSV stage
of convert_npy.py.

Founded in 2026-07-25
Modified in 2026-07-25
@author: yinlb
"""

import os
import sys
import typing
from pathlib import Path

import arrow
import numpy as np
import pandas as pd

# Allow running this script directly by file path: put the project root
# on sys.path so that `from src import utils` works.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import utils

# Observation element list (8 wind elements), identical to OBS_COLUMNS in
# convert_npy.py. Column names are kept in Chinese for consistency with
# the original observation CSV files.
OBS_COLUMNS = [
    '区站号',
    '2分钟平均风向(角度)',
    '2分钟平均风速',
    '10分钟平均风向(角度)',
    '10分钟平均风速',
    '小时内最大风速的风向',
    '最大风速',
    '极大风速的风向(角度)',
    '极大风速',
]

# Mapping from raw English columns to target Chinese columns. The raw data
# has no hourly maximum wind pair, so those two target columns stay NaN.
COLUMN_MAP = {
    'Station_Id_C': '区站号',
    'WIN_D_Avg_2mi': '2分钟平均风向(角度)',
    'WIN_S_Avg_2mi': '2分钟平均风速',
    'WIN_D_Avg_10mi': '10分钟平均风向(角度)',
    'WIN_S_Avg_10mi': '10分钟平均风速',
    'WIN_D_INST_Max': '极大风速的风向(角度)',
    'WIN_S_Inst_Max': '极大风速',
}

# Raw columns to read from each station CSV (Datetime is the time key).
RAW_COLUMNS = ['Datetime'] + list(COLUMN_MAP.keys())

# Number of hourly lead times per sample, same as convert_npy.py.
N_LEAD_HOURS = 85


def _clean_pair(
    direction: pd.Series, speed: pd.Series
) -> typing.Tuple[pd.Series, pd.Series]:
    """
    Clean one wind direction/speed pair.

    Wind direction 360 is normalized to 0. Directions outside [0, 360)
    and speeds outside [0, 1000) are treated as invalid and set to NaN.

    Args:
        direction (pd.Series): Wind direction in degrees.
        speed (pd.Series): Wind speed in m/s.

    Returns:
        typing.Tuple[pd.Series, pd.Series]: Cleaned copies of the
            direction and speed series.
    """
    direction = direction.copy()
    speed = speed.copy()
    direction[direction == 360] = 0
    direction[(direction < 0) | (direction >= 360)] = np.nan
    speed[(speed < 0) | (speed >= 1000)] = np.nan
    return direction, speed


def _load_station(file_path: Path) -> pd.DataFrame:
    """
    Load and clean one raw station CSV file.

    Read only the required columns, clean the three wind direction/speed
    pairs, and fill the unavailable hourly maximum wind pair with NaN.

    Args:
        file_path (Path): Path of the raw station CSV file.

    Returns:
        pd.DataFrame: Long-table DataFrame with a Datetime column and the
            8 target observation columns.
    """
    df = pd.read_csv(file_path, usecols=RAW_COLUMNS, low_memory=False)
    df.rename(columns=COLUMN_MAP, inplace=True)
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
    # Normalize station IDs to integers; pandas may infer different dtypes
    # (int or str) across files, and non-numeric IDs are dropped.
    df['区站号'] = pd.to_numeric(df['区站号'], errors='coerce')
    df.dropna(subset=['区站号'], inplace=True)
    df['区站号'] = df['区站号'].astype(np.int64)
    # Clean each direction/speed pair with the same validity ranges.
    for dir_col, spd_col in (
        ('2分钟平均风向(角度)', '2分钟平均风速'),
        ('10分钟平均风向(角度)', '10分钟平均风速'),
        ('极大风速的风向(角度)', '极大风速'),
    ):
        df[dir_col], df[spd_col] = _clean_pair(
            direction=df[dir_col], speed=df[spd_col]
        )
    # The raw data has no hourly maximum wind pair; fill with NaN.
    df['小时内最大风速的风向'] = np.nan
    df['最大风速'] = np.nan
    return df.loc[:, ['Datetime'] + OBS_COLUMNS]


def _build_year_array(
    obs_all: pd.DataFrame,
    station_ids: typing.List[int],
    year: int,
) -> np.ndarray:
    """
    Assemble one year of observations into a numpy array.

    The array layout matches the observation part of convert_npy.py:
    one sample every 12 hours (00/12 UTC), 85 hourly lead times per
    sample, stations ordered by ascending station ID.

    Args:
        obs_all (pd.DataFrame): Cleaned long-table observations indexed
            by (Datetime, station ID).
        station_ids (typing.List[int]): Ascending station ID list.
        year (int): Year to process.

    Returns:
        np.ndarray: Array with shape
            (sample_count, 85, station_count, 8), NaN-filled.
    """
    time_begin = arrow.get(str(year))
    time_end = time_begin.shift(years=1)
    # One sample every 12 hours (EC forecast initial times).
    sample_count = round(
        (time_end - time_begin).total_seconds() / 3600 / 12
    )
    station_count = len(station_ids)
    obs_data = np.zeros(
        (sample_count, N_LEAD_HOURS, station_count, 8), dtype=np.float32
    ) + np.nan
    for sample_idx in range(sample_count):
        for lead_time in range(N_LEAD_HOURS):
            valid_time = time_begin.shift(
                hours=12 * sample_idx + lead_time
            ).datetime
            if valid_time not in obs_all.index.levels[0]:
                continue
            hour_df = obs_all.loc[valid_time].reindex(station_ids)
            obs_data[sample_idx, lead_time, :, :] = hour_df.to_numpy(
                dtype=np.float32
            )
    return obs_data


def main(
    input_path: str,
    output_path: str,
    suffix: str = '',
) -> None:
    """
    Convert raw per-station CSVs into observation npy files.

    Iterate over all station CSV files in the input directory (filename
    is the station ID, e.g. 57469.csv), clean the wind elements, and
    save one ob{year}.npy per year in the layout produced by
    convert_npy.py.

    Args:
        input_path: Directory of raw per-station CSV files.
        output_path: Output directory for ob{year}.npy files.
        suffix: Optional output filename suffix, e.g. 'dth' produces
            ob2017_dth.npy. Defaults to '' (ob2017.npy).

    Returns:
        None
    """
    # ---- 1. Collect all station files from the input directory ----
    input_dir = Path(input_path)
    output_dir = Path(output_path)
    os.makedirs(output_dir, exist_ok=True)
    station_files = sorted(input_dir.glob('*.csv'))
    print(f'Found {len(station_files)} station files in {input_dir}')

    # ---- 2. Load and merge all stations into one long table ----
    df_list = list()
    for file_path in station_files:
        df_list.append(_load_station(file_path=file_path))
    if not df_list:
        print('No valid station data found, nothing to write')
        return
    obs_all = pd.concat(df_list, ignore_index=True)
    print(f'Loaded {len(obs_all)} records in total')

    # ---- 3. Index by (hour, station) for fast lookup ----
    obs_all['Datetime'] = obs_all['Datetime'].dt.floor('h')
    obs_all.drop_duplicates(
        subset=['Datetime', '区站号'], keep='first', inplace=True
    )
    station_ids = sorted(obs_all['区站号'].unique().tolist())
    print(f'Parsed {len(station_ids)} stations from file contents')
    obs_all.set_index(['Datetime', '区站号'], inplace=True)
    obs_all.sort_index(inplace=True)

    # ---- 4. Assemble and save one npy array per year ----
    if suffix:
        name_fmt = 'ob{:d}_' + suffix + '.npy'
    else:
        name_fmt = 'ob{:d}.npy'
    years = sorted(obs_all.index.get_level_values('Datetime').year.unique())
    for year in years:
        obs_data = _build_year_array(
            obs_all=obs_all, station_ids=station_ids, year=int(year)
        )
        filename = name_fmt.format(int(year))
        np.save(file=output_dir / filename, arr=obs_data)
        print(f'Saved {filename}, shape {obs_data.shape}')


if __name__ == '__main__':
    print('Program extract_obs.py started')
    total_start = arrow.now()

    if len(sys.argv) == 1:
        cfg = utils.load_config()
        paths = cfg['paths']
        main(
            input_path=paths['sta_raw'],
            output_path=paths['wind_data'],
            suffix=cfg['constants']['obs_suffix'],
        )
    elif len(sys.argv) in (3, 4):
        main(
            input_path=sys.argv[1],
            output_path=sys.argv[2],
            suffix=sys.argv[3] if len(sys.argv) == 4 else '',
        )

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(
        'Program extract_obs.py finished, total time '
        f'{utils.format_time(total_elapsed)}'
    )
