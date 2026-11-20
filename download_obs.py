#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMADaaS station observation download module

Download hourly surface observations for national surface stations from
CMADaaS and write one CSV per station, producing the raw per-station CSV
files consumed by extract_obs.py.

Each request covers a configurable station batch for one half-year, with
all stations in one batch by default. Responses are split by station and
appended to per-station temporary CSV files. After every request and write
succeeds, the temporary files atomically replace the final CSV files.
Temporary files are removed at startup and on exit. Keyboard interruptions
keep them for inspection.

Founded in 2026-07-29
Modified in 2026-11-20
@author: yinlb
"""

import argparse
import os
import pathlib
import sys
import typing

import arrow
import pandas as pd
from meteva import base as meb
from nmc_met_io import retrieve_cmadaas as rc

from src import utils

try:
    from tqdm import tqdm
except ImportError:
    # Fallback to a silent iterator when tqdm is unavailable.
    tqdm = None

# Number of failed station IDs printed at the end of a run; the full list is
# usually too long to be readable in a terminal.
FAILED_PREVIEW = 10

# CMADaaS elements and their quality-control columns. Each element is
# immediately followed by its quality-control column. extract_obs.py reads
# only the wind columns listed in RAW_COLUMNS/COLUMN_MAP; the additional
# columns are retained in the raw CSV files.
ELEMENTS = (
    'Station_Id_C,Datetime,'
    'PRS,Q_PRS,'
    'PRS_Sea,Q_PRS_Sea,'
    'WIN_S_Max,Q_WIN_S_Max,'
    'WIN_S_Max_OTime,Q_WIN_S_Max_OTime,'
    'WIN_D_INST_Max,Q_WIN_D_INST_Max,'
    'WIN_S_Inst_Max,Q_WIN_S_Inst_Max,'
    'WIN_S_INST_Max_OTime,Q_WIN_S_INST_Max_OTime,'
    'WIN_D_Avg_2mi,Q_WIN_D_Avg_2mi,'
    'WIN_S_Avg_2mi,Q_WIN_S_Avg_2mi,'
    'WIN_D_Avg_10mi,Q_WIN_D_Avg_10mi,'
    'WIN_S_Avg_10mi,Q_WIN_S_Avg_10mi,'
    'TEM,Q_TEM,'
    'DPT,Q_DPT,'
    'TEM_Max,Q_TEM_Max,'
    'TEM_Max_OTime,Q_TEM_Max_OTime,'
    'TEM_Min,Q_TEM_Min,'
    'TEM_Min_OTime,Q_TEM_Min_OTime,'
    'RHU,Q_RHU,'
    'RHU_Min,Q_RHU_Min,'
    'RHU_Min_OTIME,Q_RHU_Min_OTIME,'
    'PRE_1h,Q_PRE_1h,'
    'VIS_HOR_1MI,Q_VIS_HOR_1MI,'
    'VIS_HOR_10MI,Q_VIS_HOR_10MI,'
    'VIS_Min,Q_VIS_Min,'
    'VIS_Min_OTime,Q_VIS_Min_OTime,'
    'VIS,Q_VIS,'
    'WEP_Now,Q_WEP_Now,'
    'Snow_Depth,Q_Snow_Depth,'
    'Snow_PRS,Q_Snow_PRS'
)

# CMADaaS data code for hourly surface observations.
DATA_CODE = 'SURF_CHN_MUL_HOR'


def _progress(total: int, description: str):
    """
    Create a progress bar driven by manual updates.

    The bar counts theoretical hourly records for station batches written
    to temporary CSVs. Its total is the number of hours in the time range
    times the number of selected stations, so missing or duplicate returned
    rows do not affect the percentage.

    Args:
        total: Theoretical number of records in the requested time range.
        description: Label printed at the left of the progress bar.

    Returns:
        A tqdm progress bar, or None when tqdm is unavailable.
    """
    if tqdm is None:
        return None
    # miniters=1 keeps every update counted instead of letting tqdm derive
    # a coarser refresh stride from mininterval. unit_scale renders large
    # record counts with a k/M suffix.
    return tqdm(
        total=total,
        desc=description,
        unit='rec',
        unit_scale=True,
        mininterval=0.5,
        miniters=1,
        dynamic_ncols=True,
        bar_format=(
            '{l_bar}{bar} {n_fmt}/{total_fmt} '
            '[{elapsed}<{remaining}] {postfix}'
        ),
    )


def _progress_write(message: str) -> None:
    """
    Print a message without breaking an active tqdm progress bar.

    Args:
        message: The message to print.
    """
    # print() and tqdm share one output line; writing through the bar keeps
    # the bar intact and pushes the message above it.
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message)


def _station_batches(
    station_ids: typing.List[str],
    batch_size: typing.Optional[int] = None,
) -> typing.List[typing.List[str]]:
    """
    Split station IDs into balanced batches, defaulting to one full batch.

    Args:
        station_ids: Station identifiers to split.
        batch_size: Maximum stations per request. None requests all stations.

    Returns:
        Station batches whose sizes differ by at most one.

    Raises:
        ValueError: If batch_size is not a positive integer.
    """
    if batch_size is not None and batch_size < 1:
        raise ValueError('The station batch size must be positive.')
    if not station_ids:
        return []
    station_count = len(station_ids)
    max_batch_size = batch_size or station_count
    batch_count = (
        station_count + max_batch_size - 1
    ) // max_batch_size
    base_size, extra_count = divmod(station_count, batch_count)
    batches = list()
    start = 0
    for index in range(batch_count):
        size = base_size + int(index < extra_count)
        batches.append(station_ids[start:start + size])
        start += size
    return batches


def _half_year_slices(
    time_start: arrow.Arrow,
    time_end: arrow.Arrow,
) -> typing.Iterator[
    typing.Tuple[int, arrow.Arrow, arrow.Arrow]
]:
    """
    Yield clipped half-year bounds for the requested time range.

    Args:
        time_start: Start of the download time range, inclusive.
        time_end: End of the download time range, exclusive.

    Yields:
        Tuples of the year, slice start, and slice stop.
    """
    for year in range(time_start.year, time_end.year + 1):
        year_start = arrow.get(f'{year}0101', 'YYYYMMDD')
        year_stop = arrow.get(f'{year + 1}0101', 'YYYYMMDD')
        if time_end <= year_start or time_start >= year_stop:
            continue
        range_start = max(time_start, year_start)
        range_stop = min(time_end, year_stop)
        mid_year = arrow.get(f'{year}0701', 'YYYYMMDD')
        bounds = (
            (range_start, min(range_stop, mid_year)),
            (max(range_start, mid_year), range_stop),
        )
        for slice_start, slice_stop in bounds:
            if slice_stop > slice_start:
                yield year, slice_start, slice_stop


def _fetch_half_year(
    station_ids: typing.List[str],
    data_code: str,
    elements: str,
    slice_start: arrow.Arrow,
    slice_stop: arrow.Arrow,
) -> typing.Optional[pd.DataFrame]:
    """
    Download one station batch for one clipped half-year slice.

    Args:
        station_ids: Station identifiers included in the request.
        data_code: CMADaaS dataset code.
        elements: Comma-separated element list.
        slice_start: Start of this request, inclusive.
        slice_stop: End of this request, exclusive.

    Returns:
        The observation DataFrame, or None when the request failed.
    """
    time_range = (
        f'[{slice_start.format("YYYYMMDDHHmmss")},'
        f'{slice_stop.format("YYYYMMDDHHmmss")})'
    )

    try:
        sta = rc.cmadaas_obs_by_time_range_and_id(
            time_range=time_range,
            data_code=data_code,
            trans_type=False,
            elements=elements,
            sta_ids=','.join(station_ids),
        )
    except Exception as error:
        _progress_write(f'Request {time_range} failed: {error}')
        return None
    if sta is None or len(sta) == 0:
        _progress_write(f'No data returned for {time_range}')
        return None
    return sta


def _append_station_csvs(
    data: pd.DataFrame,
    output_dir: pathlib.Path,
    selected_ids: typing.Set[str],
    written_ids: typing.Set[str],
) -> None:
    """
    Split one response and append it to per-station temporary CSV files.

    The first write for a station in each run creates a temporary file and
    includes the header. Later responses append without a header.

    Args:
        data: Station-batch observation DataFrame from one response.
        output_dir: Directory containing per-station CSV files.
        selected_ids: Station identifiers requested in this run.
        written_ids: Identifiers already written in this run, updated in
            place after each successful station write.
    """
    for station_id, group in data.groupby('Station_Id_C', sort=True):
        sid = str(station_id).strip()
        if sid not in selected_ids:
            _progress_write(f'Ignore unexpected station {sid}')
            continue
        station_data = group.drop_duplicates(
            subset=['Datetime'], keep='first'
        ).sort_values(by=['Datetime'])
        temp_path = output_dir / f'{sid}.csv.tmp'
        is_new = sid not in written_ids
        station_data.to_csv(
            path_or_buf=temp_path,
            index=False,
            mode='w' if is_new else 'a',
            header=is_new,
        )
        written_ids.add(sid)


def _cleanup_temp_csvs(output_dir: pathlib.Path) -> None:
    """
    Remove per-station temporary CSV files from the output directory.

    Args:
        output_dir: Directory containing per-station CSV files.
    """
    for temp_path in output_dir.glob('*.csv.tmp'):
        temp_path.unlink()


def _finalize_station_csvs(
    output_dir: pathlib.Path,
    station_ids: typing.Iterable[str],
) -> typing.List[str]:
    """
    Replace final station CSVs with their completed temporary files.

    Args:
        output_dir: Directory containing per-station CSV files.
        station_ids: Station identifiers with completed temporary files.

    Returns:
        Station identifiers whose temporary files could not be renamed.
    """
    failed_ids = list()
    for sid in sorted(station_ids):
        temp_path = output_dir / f'{sid}.csv.tmp'
        csv_path = output_dir / f'{sid}.csv'
        try:
            os.replace(temp_path, csv_path)
        except OSError as error:
            _progress_write(f'Finalize station {sid} failed: {error}')
            failed_ids.append(sid)
    return failed_ids


def main(
    output_path: str,
    time_start: arrow.Arrow,
    time_end: arrow.Arrow,
    station_info_path: typing.Optional[str] = None,
    batch_size: typing.Optional[int] = None,
) -> typing.List[str]:
    """
    Download per-station hourly observations from CMADaaS.

    Read station IDs from a station information CSV file or from the meteva
    built-in national station list. Request balanced station batches for
    each half-year, defaulting to all stations in one request. Split each
    response by station and append to temporary CSVs. Existing final files
    are replaced only after every request and write succeeds. Stations
    absent from every response are reported as failed.

    The downloaded years are derived from the time range alone; the years
    configured in config.yaml are not consulted.

    Args:
        output_path: Output directory for per-station CSV files.
        time_start: Start of the download time range, inclusive.
        time_end: End of the download time range, exclusive.
        station_info_path: Station information CSV path containing the
            '台站号' column. If None, use meb.station_国家站. Both modes use
            SURF_CHN_MUL_HOR and the full ELEMENTS list.
        batch_size: Maximum stations per request. None requests all stations.

    Returns:
        List of station IDs that could not be downloaded or finalized.

    Raises:
        ValueError: If the time range or batch size is invalid.
    """
    # ---- 1. Resolve the requested time range ----
    if time_end <= time_start:
        raise ValueError(
            f'The end of the time range {time_end} must be later than '
            f'the start {time_start}.'
        )
    start_time = time_start.format('YYYYMMDDHHmmss')
    end_time = time_end.format('YYYYMMDDHHmmss')
    print(f'Download time range: [{start_time}, {end_time})')
    time_slices = list(_half_year_slices(time_start, time_end))
    target_years = sorted({year for year, _, _ in time_slices})
    print(f'Download years: {target_years}')

    # ---- 2. Read station IDs and use the common hourly dataset ----
    if station_info_path is None:
        station_df = meb.read_station(meb.station_国家站)
        station_column = 'id'
        station_source = 'meteva national station list'
    else:
        station_df = pd.read_csv(
            station_info_path,
            encoding='gb2312',
            low_memory=False,
        )
        station_column = '台站号'
        station_source = f'station file {station_info_path}'
    data_code = DATA_CODE
    elements = ELEMENTS
    sid_list = station_df[station_column].astype(str).tolist()
    selected_ids = set(sid_list)
    station_batches = _station_batches(sid_list, batch_size)
    max_batch_size = max(
        (len(batch) for batch in station_batches), default=0
    )
    print(f'Loaded {len(sid_list)} stations from {station_source}')
    print(f'CMADaaS data code: {data_code}')
    print(
        f'Station batches: {len(station_batches)} '
        f'(balanced, at most {max_batch_size} stations each)'
    )

    # ---- 3. Download station batches and append per-station temp CSVs ----
    output_dir = pathlib.Path(output_path)
    os.makedirs(output_dir, exist_ok=True)
    # The total and every update use theoretical hourly records. Each
    # half-year update is clipped to the requested time range.
    range_hours = int((time_end - time_start).total_seconds() // 3600)
    record_total = len(sid_list) * range_hours
    print(
        f'Estimated records: {record_total} '
        f'({len(sid_list)} station(s) x {range_hours} hour(s))'
    )
    _cleanup_temp_csvs(output_dir)
    record_bar = None
    keep_temp = False
    try:
        written_ids = set()
        record_bar = _progress(
            total=record_total, description='Downloading'
        )
        download_ok = True
        for year, slice_start, slice_stop in time_slices:
            request_hours = int(
                (slice_stop - slice_start).total_seconds() // 3600
            )
            for station_ids in station_batches:
                sta = _fetch_half_year(
                    station_ids=station_ids,
                    data_code=data_code,
                    elements=elements,
                    slice_start=slice_start,
                    slice_stop=slice_stop,
                )
                if sta is None or len(sta) == 0:
                    download_ok = False
                    break
                try:
                    _append_station_csvs(
                        data=sta,
                        output_dir=output_dir,
                        selected_ids=selected_ids,
                        written_ids=written_ids,
                    )
                except Exception as error:
                    _progress_write(
                        f'Save year {year} data failed: {error}'
                    )
                    download_ok = False
                else:
                    if record_bar is not None:
                        record_bar.update(
                            len(station_ids) * request_hours
                        )
                        record_bar.set_postfix_str(
                            str(year), refresh=False
                        )
                # Release this response before starting the next request.
                del sta
                if not download_ok:
                    break
            if not download_ok:
                break

        # ---- 4. Finalize completed station CSVs ----
        if download_ok:
            failed_ids = sorted(selected_ids - written_ids)
            failed_ids.extend(
                _finalize_station_csvs(
                    output_dir=output_dir,
                    station_ids=written_ids,
                )
            )
        else:
            failed_ids = sid_list.copy()
    except KeyboardInterrupt:
        keep_temp = True
        raise
    finally:
        try:
            if record_bar is not None:
                record_bar.close()
        finally:
            if not keep_temp:
                _cleanup_temp_csvs(output_dir)

    # ---- 5. Report incomplete stations ----
    if failed_ids:
        # A long failure list is unreadable; show only the first few IDs.
        preview = ', '.join(failed_ids[:FAILED_PREVIEW])
        suffix = ', ...' if len(failed_ids) > FAILED_PREVIEW else ''
        print(
            f'{len(failed_ids)} station(s) failed: {preview}{suffix}'
        )
    else:
        print('All stations downloaded successfully')
    return failed_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download CMADaaS observations to per-station CSV files.'
    )
    parser.add_argument(
        '--national',
        action='store_true',
        help='Use the meteva national station list.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        metavar='N',
        help='Maximum stations per request (default: all stations).',
    )
    parser.add_argument(
        'positionals',
        nargs='+',
        help='[output_path] time_start time_end, using YYYYMMDD dates.',
    )
    cli_args = parser.parse_args()
    positionals = cli_args.positionals
    if len(positionals) not in (2, 3):
        parser.error('Expected [output_path] time_start time_end.')
    if cli_args.batch_size is not None and cli_args.batch_size < 1:
        parser.error('--batch-size must be a positive integer.')

    print('Program download_obs.py started')
    total_start = arrow.now()
    cfg = utils.load_config()
    station_info_path = (
        None if cli_args.national else cfg['paths']['station_csv']
    )
    output_path = (
        positionals[0]
        if len(positionals) == 3
        else cfg['paths']['sta_raw']
    )
    failed = main(
        output_path=output_path,
        station_info_path=station_info_path,
        time_start=arrow.get(positionals[-2], 'YYYYMMDD'),
        time_end=arrow.get(positionals[-1], 'YYYYMMDD'),
        batch_size=cli_args.batch_size,
    )

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(
        'Program download_obs.py finished, total time '
        f'{utils.format_time(total_elapsed)}'
    )
    if failed:
        sys.exit(1)
