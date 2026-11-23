#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download CMADaaS observations with resumable per-station CSV output.

Requests remain serial and use configurable station batches for clipped
half-year slices. Completed station slices are checkpointed atomically. Empty
or failed requests stay unresolved, temporary CSVs are preserved, and other
batches continue. Stations are finalized independently after all their slices
complete; stations without any observations receive a header-only CSV.

Founded in 2026-07-29
Modified in 2026-11-23
@author: yinlb
"""

import argparse
import contextlib
import hashlib
import io
import json
import logging
import os
import pathlib
import time
import typing
from logging.handlers import RotatingFileHandler

import arrow
import pandas as pd
from meteva import base as meb
from nmc_met_io import retrieve_cmadaas as rc

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

FAILED_PREVIEW = 10
STATE_NAME = '.download_state.json'
LOG_NAME = 'download.log'
STATE_VERSION = 1
LOG_MAX_BYTES = 5 * 1024 * 1024

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
DATA_CODE = 'SURF_CHN_MUL_HOR'
CSV_COLUMNS = tuple(ELEMENTS.split(','))


class FetchResult(typing.NamedTuple):
    """Result of one serial CMADaaS request."""

    status: str
    data: typing.Optional[pd.DataFrame]
    message: str


def _progress(total: int, description: str):
    """Create a progress bar based on completed theoretical station hours."""
    if tqdm is None:
        return None
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
    """Print one message without breaking the active progress bar."""
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message)


def _station_batches(
    station_ids: typing.List[str],
    batch_size: typing.Optional[int] = None,
) -> typing.List[typing.List[str]]:
    """Split station IDs into balanced batches."""
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
) -> typing.Iterator[typing.Tuple[int, arrow.Arrow, arrow.Arrow]]:
    """Yield clipped half-year bounds for the requested time range."""
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


def _slice_key(slice_start: arrow.Arrow, slice_stop: arrow.Arrow) -> str:
    """Return the stable key of one clipped time slice."""
    return '{0}-{1}'.format(
        slice_start.format('YYYYMMDDHHmmss'),
        slice_stop.format('YYYYMMDDHHmmss'),
    )


def _fetch_half_year(
    station_ids: typing.List[str],
    data_code: str,
    elements: str,
    slice_start: arrow.Arrow,
    slice_stop: arrow.Arrow,
) -> FetchResult:
    """Fetch one request and retain the raw text emitted by the client."""
    time_range = (
        f'[{slice_start.format("YYYYMMDDHHmmss")},'
        f'{slice_stop.format("YYYYMMDDHHmmss")})'
    )
    raw_output = io.StringIO()
    try:
        with contextlib.redirect_stdout(raw_output):
            sta = rc.cmadaas_obs_by_time_range_and_id(
                time_range=time_range,
                data_code=data_code,
                trans_type=False,
                elements=elements,
                sta_ids=','.join(station_ids),
            )
    except Exception as error:
        return FetchResult('error', None, str(error))
    details = raw_output.getvalue().strip()
    if sta is None or len(sta) == 0:
        message = details or 'CMADaaS returned no records'
        return FetchResult('empty', None, message)
    return FetchResult('ok', sta, details)


def _append_station_csvs(
    data: pd.DataFrame,
    output_dir: pathlib.Path,
    selected_ids: typing.Set[str],
) -> typing.Dict[str, int]:
    """Append one response and return updated temporary file sizes."""
    sizes = {}
    for station_id, group in data.groupby('Station_Id_C', sort=True):
        sid = str(station_id).strip()
        if sid not in selected_ids:
            _progress_write(f'Ignore unexpected station {sid}')
            continue
        station_data = group.drop_duplicates(
            subset=['Datetime'], keep='first'
        ).sort_values(by=['Datetime'])
        temp_path = output_dir / f'{sid}.csv.tmp'
        is_new = not temp_path.exists() or temp_path.stat().st_size == 0
        station_data.to_csv(
            path_or_buf=temp_path,
            index=False,
            mode='w' if is_new else 'a',
            header=is_new,
        )
        sizes[sid] = temp_path.stat().st_size
    return sizes


def _job_signature(
    station_ids: typing.List[str],
    time_start: arrow.Arrow,
    time_end: arrow.Arrow,
) -> str:
    """Hash all job settings that must match before a state is reused."""
    content = {
        'station_ids': station_ids,
        'time_start': time_start.format('YYYYMMDDHHmmss'),
        'time_end': time_end.format('YYYYMMDDHHmmss'),
        'data_code': DATA_CODE,
        'elements': ELEMENTS,
        'version': STATE_VERSION,
    }
    raw = json.dumps(
        content, ensure_ascii=False, sort_keys=True
    ).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _new_state(signature: str) -> dict:
    """Create an empty resumable job state."""
    return {
        'version': STATE_VERSION,
        'signature': signature,
        'stations': {},
        'failures': {},
    }


def _load_state(
    output_dir: pathlib.Path, signature: str, restart: bool
) -> dict:
    """Load a matching state or explicitly reset incomplete output."""
    state_path = output_dir / STATE_NAME
    if restart:
        for temp_path in output_dir.glob('*.csv.tmp'):
            temp_path.unlink()
        if state_path.exists():
            state_path.unlink()
        return _new_state(signature)
    if not state_path.exists():
        leftovers = list(output_dir.glob('*.csv.tmp'))
        if leftovers:
            raise RuntimeError(
                'Untracked temporary CSVs exist without a state file; '
                'inspect them or run with --restart.'
            )
        return _new_state(signature)
    with open(state_path, 'r', encoding='utf-8') as file_handle:
        state = json.load(file_handle)
    if state.get('signature') != signature:
        raise ValueError(
            'The existing download state belongs to another job; '
            'use --restart or another output directory.'
        )
    return state


def _station_state(state: dict, station_id: str) -> dict:
    """Return the mutable checkpoint of one station."""
    return state['stations'].setdefault(
        station_id,
        {'done_slices': [], 'temp_bytes': 0, 'status': 'pending'},
    )


def _repair_temp_csvs(output_dir: pathlib.Path, state: dict) -> None:
    """Roll back writes that occurred after the last atomic checkpoint."""
    for sid, station in state['stations'].items():
        if station.get('status') in ('complete', 'no_data'):
            continue
        expected = int(station.get('temp_bytes', 0))
        temp_path = output_dir / f'{sid}.csv.tmp'
        if not temp_path.exists():
            if expected:
                raise RuntimeError(
                    f'Temporary CSV is missing for station {sid}; '
                    'restore it or run with --restart.'
                )
            continue
        actual = temp_path.stat().st_size
        if actual > expected:
            with open(temp_path, 'r+b') as file_handle:
                file_handle.truncate(expected)
        elif actual < expected:
            raise RuntimeError(
                f'Temporary CSV is shorter than its checkpoint for {sid}; '
                'restore it or run with --restart.'
            )


def _save_state(output_dir: pathlib.Path, state: dict) -> None:
    """Atomically replace the current state without event-history growth."""
    state['updated_at'] = arrow.now().isoformat()
    path = output_dir / STATE_NAME
    temp_path = output_dir / (STATE_NAME + '.tmp')
    with open(temp_path, 'w', encoding='utf-8') as file_handle:
        json.dump(state, file_handle, ensure_ascii=False, indent=1)
    os.replace(temp_path, path)


def _request_key(slice_key: str, station_ids: typing.List[str]) -> str:
    """Identify one exact request for failure reporting."""
    raw = slice_key + ':' + ','.join(station_ids)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def _prune_failures(state: dict) -> None:
    """Drop failure records whose station slices have since completed."""
    for key, failure in list(state['failures'].items()):
        slice_key = failure['slice_key']
        resolved = True
        for sid in failure['station_ids']:
            station = _station_state(state, sid)
            if slice_key == 'finalize':
                resolved = station.get('status') in ('complete', 'no_data')
            else:
                resolved = slice_key in station['done_slices']
            if not resolved:
                break
        if resolved:
            state['failures'].pop(key, None)


def _record_failure(
    state: dict,
    slice_key: str,
    station_ids: typing.List[str],
    message: str,
    attempts: int,
) -> None:
    """Store the latest unresolved failure once."""
    key = _request_key(slice_key, station_ids)
    state['failures'][key] = {
        'slice_key': slice_key,
        'station_ids': station_ids,
        'message': message,
        'attempts': attempts,
        'updated_at': arrow.now().isoformat(),
    }


def _finalize_ready_stations(
    output_dir: pathlib.Path,
    state: dict,
    station_ids: typing.List[str],
    slice_keys: typing.Set[str],
) -> None:
    """Publish each station as soon as all its slices are complete."""
    for sid in station_ids:
        station = _station_state(state, sid)
        if station.get('status') in ('complete', 'no_data'):
            continue
        if not slice_keys.issubset(set(station['done_slices'])):
            station['status'] = (
                'partial' if station.get('temp_bytes') else 'pending'
            )
            continue
        temp_path = output_dir / f'{sid}.csv.tmp'
        csv_path = output_dir / f'{sid}.csv'
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            pd.DataFrame(columns=CSV_COLUMNS).to_csv(temp_path, index=False)
            status = 'no_data'
        else:
            status = 'complete'
        try:
            os.replace(temp_path, csv_path)
        except OSError as error:
            station['status'] = 'partial'
            _record_failure(
                state,
                'finalize',
                [sid],
                str(error),
                1,
            )
            continue
        station['status'] = status
        station['temp_bytes'] = 0


def _configure_logger(output_dir: pathlib.Path):
    """Configure one bounded log file and return its handler."""
    logger = logging.getLogger('download_obs')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = RotatingFileHandler(
        output_dir / LOG_NAME,
        maxBytes=LOG_MAX_BYTES,
        backupCount=1,
        encoding='utf-8',
    )
    handler.setFormatter(
        logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    )
    logger.addHandler(handler)
    return logger, handler


def _report(logger: logging.Logger, level: int, message: str) -> None:
    """Emit a message once to the console and once to the bounded log."""
    _progress_write(message)
    logger.log(level, message)


def main(
    output_path: str,
    time_start: arrow.Arrow,
    time_end: arrow.Arrow,
    station_info_path: typing.Optional[str] = None,
    batch_size: typing.Optional[int] = None,
    restart: bool = False,
    max_attempts: int = 2,
    retry_wait: float = 60.0,
) -> typing.List[str]:
    """
    Download observations serially and return only incomplete station IDs.

    Existing temporary files and matching checkpoints are reused. Explicit
    errors may be retried after ``retry_wait`` seconds; empty responses are
    saved for a later resume instead of being retried immediately.
    """
    if time_end <= time_start:
        raise ValueError('The end time must be later than the start time.')
    if max_attempts < 1:
        raise ValueError('max_attempts must be positive.')
    if retry_wait < 0:
        raise ValueError('retry_wait must be non-negative.')

    time_slices = list(_half_year_slices(time_start, time_end))
    target_years = sorted({year for year, _, _ in time_slices})
    print(
        'Download time range: [{0}, {1})'.format(
            time_start.format('YYYYMMDDHHmmss'),
            time_end.format('YYYYMMDDHHmmss'),
        )
    )
    print(f'Download years: {target_years}')

    if station_info_path is None:
        station_df = meb.read_station(meb.station_国家站)
        station_column = 'id'
        station_source = 'meteva national station list'
    else:
        station_df = pd.read_csv(
            station_info_path,
            encoding='gb2312',
            low_memory=False,
            dtype={'台站号': str},
        )
        station_column = '台站号'
        station_source = f'station file {station_info_path}'
    sid_list = station_df[station_column].astype(str).tolist()
    station_batches = _station_batches(sid_list, batch_size)
    print(f'Loaded {len(sid_list)} stations from {station_source}')
    print(f'CMADaaS data code: {DATA_CODE}')
    print(
        f'Station batches: {len(station_batches)} '
        f'(balanced, at most {batch_size or len(sid_list)} stations each)'
    )

    output_dir = pathlib.Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger, handler = _configure_logger(output_dir)
    signature = _job_signature(sid_list, time_start, time_end)
    state = _load_state(output_dir, signature, restart)
    for sid in sid_list:
        _station_state(state, sid)
    _repair_temp_csvs(output_dir, state)
    _save_state(output_dir, state)
    slice_keys = {
        _slice_key(start, stop) for _, start, stop in time_slices
    }
    slice_hours = {
        _slice_key(start, stop): int(
            (stop - start).total_seconds() // 3600
        )
        for _, start, stop in time_slices
    }
    range_hours = int((time_end - time_start).total_seconds() // 3600)
    record_total = len(sid_list) * range_hours
    completed_records = sum(
        slice_hours.get(key, 0)
        for sid in sid_list
        for key in _station_state(state, sid)['done_slices']
    )
    print(
        f'Estimated records: {record_total} '
        f'({len(sid_list)} station(s) x {range_hours} hour(s))'
    )
    _report(
        logger,
        logging.INFO,
        f'Run started: stations={len(sid_list)} slices={len(time_slices)} '
        f'batches={len(station_batches)} restart={restart}',
    )

    record_bar = _progress(record_total, 'Downloading')
    if record_bar is not None and completed_records:
        record_bar.update(completed_records)
    fetch_seconds = 0.0
    write_seconds = 0.0
    try:
        for batch_index, batch in enumerate(station_batches, 1):
            for year, slice_start, slice_stop in time_slices:
                key = _slice_key(slice_start, slice_stop)
                pending = [
                    sid for sid in batch
                    if key not in _station_state(state, sid)['done_slices']
                ]
                if not pending:
                    continue
                result = FetchResult('error', None, 'No request attempted')
                attempts = 0
                for attempts in range(1, max_attempts + 1):
                    started = time.perf_counter()
                    result = _fetch_half_year(
                        pending, DATA_CODE, ELEMENTS,
                        slice_start, slice_stop,
                    )
                    fetch_seconds += time.perf_counter() - started
                    if result.status != 'error':
                        break
                    if attempts < max_attempts:
                        _report(
                            logger,
                            logging.WARNING,
                            f'Retrying error: year={year} '
                            f'batch={batch_index}/{len(station_batches)} '
                            f'attempt={attempts}/{max_attempts} '
                            f'stations={len(pending)}',
                        )
                        time.sleep(retry_wait)
                if result.status != 'ok':
                    message = (
                        f'Request incomplete: year={year} '
                        f'batch={batch_index}/{len(station_batches)} '
                        f'status={result.status} stations={len(pending)} '
                        f'attempts={attempts} detail={result.message}'
                    )
                    _record_failure(
                        state, key, pending, result.message, attempts
                    )
                    _save_state(output_dir, state)
                    _report(logger, logging.ERROR, message)
                    continue

                before_sizes = {
                    sid: int(_station_state(state, sid).get('temp_bytes', 0))
                    for sid in pending
                }
                try:
                    started = time.perf_counter()
                    sizes = _append_station_csvs(
                        result.data, output_dir, set(pending)
                    )
                    write_seconds += time.perf_counter() - started
                except Exception as error:
                    for sid, size in before_sizes.items():
                        temp_path = output_dir / f'{sid}.csv.tmp'
                        if temp_path.exists():
                            with open(temp_path, 'r+b') as file_handle:
                                file_handle.truncate(size)
                    _record_failure(
                        state, key, pending, str(error), attempts
                    )
                    _save_state(output_dir, state)
                    _report(
                        logger,
                        logging.ERROR,
                        f'Write failed: year={year} batch={batch_index} '
                        f'stations={len(pending)} detail={error}',
                    )
                    continue

                for sid in pending:
                    station = _station_state(state, sid)
                    station['done_slices'].append(key)
                    station['temp_bytes'] = sizes.get(
                        sid, int(station.get('temp_bytes', 0))
                    )
                    station['status'] = 'downloading'
                _prune_failures(state)
                _save_state(output_dir, state)
                if record_bar is not None:
                    record_bar.update(len(pending) * slice_hours[key])
                    record_bar.set_postfix_str(
                        f'{year} b{batch_index}/{len(station_batches)}',
                        refresh=False,
                    )

            _finalize_ready_stations(
                output_dir, state, batch, slice_keys
            )
            _prune_failures(state)
            _save_state(output_dir, state)
            complete = sum(
                _station_state(state, sid).get('status')
                in ('complete', 'no_data')
                for sid in batch
            )
            _report(
                logger,
                logging.INFO,
                f'Batch finished: batch={batch_index}/{len(station_batches)} '
                f'complete={complete}/{len(batch)}',
            )
    except KeyboardInterrupt:
        _save_state(output_dir, state)
        _report(logger, logging.WARNING, 'Interrupted; checkpoints preserved')
        raise
    finally:
        if record_bar is not None:
            record_bar.close()
        logger.removeHandler(handler)
        handler.close()

    incomplete = sorted(
        sid for sid in sid_list
        if _station_state(state, sid).get('status')
        not in ('complete', 'no_data')
    )
    complete = sum(
        _station_state(state, sid).get('status') == 'complete'
        for sid in sid_list
    )
    no_data = sum(
        _station_state(state, sid).get('status') == 'no_data'
        for sid in sid_list
    )
    print(
        f'Download summary: complete={complete} no_data={no_data} '
        f'incomplete={len(incomplete)} '
        f'fetch_seconds={fetch_seconds:.1f} '
        f'write_seconds={write_seconds:.1f}'
    )
    if incomplete:
        preview = ', '.join(incomplete[:FAILED_PREVIEW])
        suffix = ', ...' if len(incomplete) > FAILED_PREVIEW else ''
        print(f'{len(incomplete)} station(s) incomplete: {preview}{suffix}')
    else:
        print('All stations downloaded successfully')
    return incomplete


def _format_elapsed(seconds: float) -> str:
    """Format total runtime without depending on legacy configuration."""
    return f'{seconds / 3600.0:.6f} hours'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download CMADaaS observations with resumable CSV output.'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--national', action='store_true',
        help='Use the meteva national station list.',
    )
    group.add_argument(
        '--station-info',
        help='Station CSV containing the 台站号 column.',
    )
    parser.add_argument('--batch-size', type=int, default=None, metavar='N')
    parser.add_argument(
        '--restart', action='store_true',
        help='Explicitly discard incomplete state and temporary CSVs.',
    )
    parser.add_argument('--max-attempts', type=int, default=2)
    parser.add_argument('--retry-wait', type=float, default=60.0)
    parser.add_argument('output_path')
    parser.add_argument('time_start')
    parser.add_argument('time_end')
    cli_args = parser.parse_args()
    if cli_args.batch_size is not None and cli_args.batch_size < 1:
        parser.error('--batch-size must be a positive integer.')
    if cli_args.max_attempts < 1:
        parser.error('--max-attempts must be positive.')
    if cli_args.retry_wait < 0:
        parser.error('--retry-wait must be non-negative.')

    print('Program download_obs.py started')
    total_start = arrow.now()
    failed = main(
        output_path=cli_args.output_path,
        station_info_path=cli_args.station_info,
        time_start=arrow.get(cli_args.time_start, 'YYYYMMDD'),
        time_end=arrow.get(cli_args.time_end, 'YYYYMMDD'),
        batch_size=cli_args.batch_size,
        restart=cli_args.restart,
        max_attempts=cli_args.max_attempts,
        retry_wait=cli_args.retry_wait,
    )
    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(
        'Program download_obs.py finished, total time '
        f'{_format_elapsed(total_elapsed)}'
    )
    if failed:
        raise SystemExit(1)
