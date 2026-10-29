#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification for the 2411-station experiment.

Loads the per-fold predictions and the aligned observations, computes the
overall scores (CC, MAE, RMSE, MRE, OA, kappa, TS) and the grouped RMSE
statistics (year, month, lead time, hour of day, station, wind direction),
and writes the results as CSV files.

Founded in 2026-10-29
Modified in 2026-10-29
@author: yinlb, deepseek-flash
"""

import argparse
import pathlib
import typing

import numpy as np
import pandas as pd

from src import data_2411

MODELS = ('ECMWF-IFS', 'PDFM', 'MLR', 'LightGBM-A', 'LightGBM-B')
FILE_NAMES = {
    'ECMWF-IFS': 'ecmwf_ifs',
    'PDFM': 'pdfm',
    'MLR': 'mlr',
    'LightGBM-A': 'lgb_a',
    'LightGBM-B': 'lgb_b',
}
# Beaufort-like wind-speed thresholds (m/s), identical to src/metrics.py.
THRES = (0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2)
N_GRADE = len(THRES) + 1
N_DIR = 8


def speed_grade(speed: np.ndarray) -> np.ndarray:
    """
    Map wind speed to a 0-8 grade; NaN becomes -1.

    Args:
        speed: Wind speed array.

    Returns:
        np.ndarray: Integer grade array.
    """
    grade = np.full(speed.shape, -1, dtype=np.int16)
    valid = ~np.isnan(speed)
    grade[valid] = 0
    for index, threshold in enumerate(THRES):
        grade[valid & (speed >= threshold)] = index + 1
    return grade


def dir_grade(direction: np.ndarray) -> np.ndarray:
    """
    Map wind direction to an 8-sector grade (0 = north); NaN becomes -1.

    Args:
        direction: Wind direction array.

    Returns:
        np.ndarray: Integer sector array.
    """
    grade = np.full(direction.shape, -1, dtype=np.int16)
    valid = (~np.isnan(direction)) & (direction != 999017)
    angle = direction[valid].astype(np.float64)
    grade[valid] = np.floor(((angle + 22.5) % 360.0) / 45.0).astype(np.int16)
    return grade


def confusion(ob: np.ndarray, pr: np.ndarray, n_cls: int) -> np.ndarray:
    """
    Build a confusion matrix from two grade arrays.

    Args:
        ob: Observed grades (-1 marks invalid).
        pr: Predicted grades (-1 marks invalid).
        n_cls: Number of classes.

    Returns:
        np.ndarray: Confusion matrix with shape (n_cls, n_cls).
    """
    mask = (ob >= 0) & (pr >= 0)
    flat = ob[mask].astype(np.int64) * n_cls + pr[mask].astype(np.int64)
    counts = np.bincount(flat, minlength=n_cls * n_cls)
    return counts.reshape(n_cls, n_cls).astype(np.float64)


def oa_kappa(cm: np.ndarray) -> typing.Tuple[float, float]:
    """
    Compute overall accuracy and Cohen's kappa from a confusion matrix.

    Args:
        cm: Confusion matrix.

    Returns:
        typing.Tuple[float, float]: OA and kappa.
    """
    total = cm.sum()
    if total == 0:
        return np.nan, np.nan
    oa = np.trace(cm) / total
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    pe = float(np.sum(row * col) / total / total)
    kappa = (oa - pe) / (1.0 - pe) if pe < 1.0 else np.nan
    return float(oa), float(kappa)


def threat_score(cm: np.ndarray) -> np.ndarray:
    """
    Compute the threat score of every speed grade.

    Args:
        cm: Speed-grade confusion matrix.

    Returns:
        np.ndarray: TS per grade (0-8).
    """
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    hit = np.diag(cm)
    denom = row + col - hit
    with np.errstate(invalid='ignore', divide='ignore'):
        ts = np.where(denom > 0, hit / denom, np.nan)
    return ts


def scores(
    ob_dir: np.ndarray, ob_spd: np.ndarray,
    pr_dir: np.ndarray, pr_spd: np.ndarray,
) -> typing.Dict[str, float]:
    """
    Compute the overall verification scores.

    Args:
        ob_dir: Observed direction.
        ob_spd: Observed speed.
        pr_dir: Predicted direction.
        pr_spd: Predicted speed.

    Returns:
        typing.Dict[str, float]: Score dictionary.
    """
    out = {}
    mask = (~np.isnan(ob_spd)) & (~np.isnan(pr_spd))
    obs = ob_spd[mask].astype(np.float64)
    pred = pr_spd[mask].astype(np.float64)
    out['ws_cc'] = float(np.corrcoef(obs, pred)[0, 1])
    out['ws_mae'] = float(np.mean(np.abs(pred - obs)))
    out['ws_rmse'] = float(np.sqrt(np.mean((pred - obs) ** 2)))
    denom = pred + obs
    keep = denom > 0
    ratio = np.abs((pred[keep] - obs[keep]) / denom[keep])
    out['ws_mre'] = float(np.mean(ratio))

    cm_ws = confusion(speed_grade(obs), speed_grade(pred), N_GRADE)
    out['ws_oa'], out['ws_kappa'] = oa_kappa(cm_ws)
    ts_val = threat_score(cm_ws)
    for grade in range(1, N_GRADE):
        out['ws_ts{0}'.format(grade)] = float(ts_val[grade])

    mask_dir = (
        (~np.isnan(ob_dir)) & (~np.isnan(pr_dir))
        & (ob_dir != 999017) & (pr_dir != 999017)
    )
    delta = (
        pr_dir[mask_dir].astype(np.float64)
        - ob_dir[mask_dir].astype(np.float64)
    )
    delta = (delta + 180.0) % 360.0 - 180.0
    out['wd_mae'] = float(np.mean(np.abs(delta)))
    out['wd_rmse'] = float(np.sqrt(np.mean(delta ** 2)))
    cm_wd = confusion(dir_grade(ob_dir), dir_grade(pr_dir), N_DIR)
    out['wd_oa'], out['wd_kappa'] = oa_kappa(cm_wd)
    return out


def load_obs(
    cfg: dict, station_ids: typing.Sequence[str], year: int
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Load the aligned observations of one held-out year.

    Args:
        cfg: Configuration dictionary.
        station_ids: Station IDs in the training order.
        year: Held-out year.

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: Direction and speed arrays
        with shape (n_samples, n_hours, n_stations).
    """
    slices = data_2411.year_slices(cfg)
    start, end = slices[year]
    n_sample = end - start
    n_station = len(station_ids)
    direction = np.full((n_sample, 24, n_station), np.nan, np.float32)
    speed = np.full((n_sample, 24, n_station), np.nan, np.float32)
    for index, sta in enumerate(station_ids):
        arrays = data_2411.load_cache(sta, cfg)
        if arrays is None:
            continue
        direction[:, :, index] = arrays['dir'][start:end]
        speed[:, :, index] = arrays['spd'][start:end]
    return direction, speed


def load_pred(
    cfg: dict, model: str, year: int
) -> typing.Optional[np.ndarray]:
    """
    Load one model prediction file.

    Args:
        cfg: Configuration dictionary.
        model: Model name.
        year: Held-out year.

    Returns:
        typing.Optional[np.ndarray]: Prediction array or None when missing.
    """
    path = (
        pathlib.Path(cfg['paths']['pred_dir'])
        / '{0}_{1}.npy'.format(FILE_NAMES[model], year)
    )
    if not path.exists():
        return None
    return np.load(path)


def time_labels(cfg: dict) -> typing.Dict[str, np.ndarray]:
    """
    Build the grouping labels shared by every fold.

    Args:
        cfg: Configuration dictionary.

    Returns:
        typing.Dict[str, np.ndarray]: Month, hour-of-day and lead-time
        labels with shape (n_samples, n_hours).
    """
    times = data_2411.sample_times(cfg)
    n_sample = int(cfg['constants']['n_samples'])
    month = times.month.to_numpy().reshape(n_sample, 24)
    hour = times.hour.to_numpy().reshape(n_sample, 24)
    lead = np.broadcast_to(np.arange(24), (n_sample, 24)).copy()
    return {'month': month, 'hour': hour, 'lead': lead}


class ScoreAcc(object):
    """Streaming accumulator for the overall verification scores."""

    def __init__(self, marker: float = 999017.0) -> None:
        """
        Initialize the accumulator.

        Args:
            marker: Calm-wind direction marker.
        """
        self.marker = float(marker)
        self.n_ws = 0
        self.s_abs = 0.0
        self.s_sq = 0.0
        self.s_mre = 0.0
        self.n_mre = 0
        self.s_o = 0.0
        self.s_p = 0.0
        self.s_oo = 0.0
        self.s_pp = 0.0
        self.s_op = 0.0
        self.cm_ws = np.zeros((N_GRADE, N_GRADE), dtype=np.float64)
        self.n_wd = 0
        self.s_wabs = 0.0
        self.s_wsq = 0.0
        self.cm_wd = np.zeros((N_DIR, N_DIR), dtype=np.float64)

    def add(
        self, ob_dir: np.ndarray, ob_spd: np.ndarray,
        pr_dir: np.ndarray, pr_spd: np.ndarray,
    ) -> None:
        """
        Add one block of matched observations and predictions.

        Args:
            ob_dir: Observed direction.
            ob_spd: Observed speed.
            pr_dir: Predicted direction.
            pr_spd: Predicted speed.
        """
        mask = (~np.isnan(ob_spd)) & (~np.isnan(pr_spd))
        obs = ob_spd[mask].astype(np.float64)
        pred = pr_spd[mask].astype(np.float64)
        if obs.size == 0:
            return
        error = pred - obs
        self.n_ws += int(obs.size)
        self.s_abs += float(np.sum(np.abs(error)))
        self.s_sq += float(np.sum(error ** 2))
        denom = pred + obs
        keep = denom > 0
        self.s_mre += float(
            np.sum(np.abs((pred[keep] - obs[keep]) / denom[keep]))
        )
        self.n_mre += int(keep.sum())
        self.s_o += float(obs.sum())
        self.s_p += float(pred.sum())
        self.s_oo += float(np.sum(obs ** 2))
        self.s_pp += float(np.sum(pred ** 2))
        self.s_op += float(np.sum(obs * pred))
        self.cm_ws += confusion(speed_grade(obs), speed_grade(pred), N_GRADE)

        mask_dir = (
            (~np.isnan(ob_dir)) & (~np.isnan(pr_dir))
            & (ob_dir != self.marker) & (pr_dir != self.marker)
        )
        delta = (
            pr_dir[mask_dir].astype(np.float64)
            - ob_dir[mask_dir].astype(np.float64)
        )
        delta = (delta + 180.0) % 360.0 - 180.0
        self.n_wd += int(delta.size)
        self.s_wabs += float(np.sum(np.abs(delta)))
        self.s_wsq += float(np.sum(delta ** 2))
        self.cm_wd += confusion(dir_grade(ob_dir), dir_grade(pr_dir), N_DIR)

    def result(self) -> typing.Dict[str, float]:
        """
        Compute the accumulated scores.

        Returns:
            typing.Dict[str, float]: Score dictionary.
        """
        out = {}
        n_ws = max(self.n_ws, 1)
        var_o = self.n_ws * self.s_oo - self.s_o ** 2
        var_p = self.n_ws * self.s_pp - self.s_p ** 2
        cov = self.n_ws * self.s_op - self.s_o * self.s_p
        denom = np.sqrt(max(var_o * var_p, 0.0))
        out['ws_cc'] = float(cov / denom) if denom > 0 else np.nan
        out['ws_mae'] = self.s_abs / n_ws
        out['ws_rmse'] = float(np.sqrt(self.s_sq / n_ws))
        out['ws_mre'] = (
            self.s_mre / self.n_mre if self.n_mre > 0 else np.nan
        )
        out['ws_oa'], out['ws_kappa'] = oa_kappa(self.cm_ws)
        ts_val = threat_score(self.cm_ws)
        for grade in range(1, N_GRADE):
            out['ws_ts{0}'.format(grade)] = float(ts_val[grade])
        n_wd = max(self.n_wd, 1)
        out['wd_mae'] = self.s_wabs / n_wd
        out['wd_rmse'] = float(np.sqrt(self.s_wsq / n_wd))
        out['wd_oa'], out['wd_kappa'] = oa_kappa(self.cm_wd)
        return out


def dir_group_rmse(
    ob_dir: np.ndarray, ob_spd: np.ndarray,
    pr_dir: np.ndarray, pr_spd: np.ndarray,
) -> np.ndarray:
    """
    Compute speed and direction RMSE per observed direction sector.

    Args:
        ob_dir: Observed direction.
        ob_spd: Observed speed.
        pr_dir: Predicted direction.
        pr_spd: Predicted speed.

    Returns:
        np.ndarray: RMSE with shape (2, 8); row 0 is speed, row 1 is
        direction.
    """
    sector = dir_grade(ob_dir)
    out = np.full((2, N_DIR), np.nan, dtype=np.float64)
    diff_spd = pr_spd - ob_spd
    delta = (
        pr_dir.astype(np.float64) - ob_dir.astype(np.float64) + 180.0
    ) % 360.0 - 180.0
    valid_dir = (
        (~np.isnan(ob_dir)) & (~np.isnan(pr_dir))
        & (ob_dir != 999017) & (pr_dir != 999017)
    )
    for index in range(N_DIR):
        mask_spd = (sector == index) & (~np.isnan(diff_spd))
        if mask_spd.any():
            out[0, index] = float(np.sqrt(np.mean(diff_spd[mask_spd] ** 2)))
        mask_dir = (sector == index) & valid_dir
        if mask_dir.any():
            out[1, index] = float(np.sqrt(np.mean(delta[mask_dir] ** 2)))
    return out


def safe_nanmean(stacked: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute a NaN-aware mean without the numpy all-NaN edge case.

    Args:
        stacked: Input array.
        axis: Axis to reduce.

    Returns:
        np.ndarray: Mean with NaN where no valid value exists.
    """
    count = np.sum(~np.isnan(stacked), axis=axis)
    total = np.nansum(stacked, axis=axis)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def group_rmse(
    obs: np.ndarray, pred: np.ndarray, label: np.ndarray, n_group: int
) -> np.ndarray:
    """
    Compute RMSE for every group of a label array.

    Args:
        obs: Observations.
        pred: Predictions with the same shape as obs.
        label: Group labels, either the same shape as obs or broadcastable
            to it.
        n_group: Number of groups.

    Returns:
        np.ndarray: RMSE per group.
    """
    diff = pred - obs
    lab = np.broadcast_to(label, diff.shape)
    out = np.full(n_group, np.nan, dtype=np.float64)
    valid = ~np.isnan(diff)
    for group in range(n_group):
        mask = (lab == group) & valid
        if mask.any():
            out[group] = float(np.sqrt(np.mean(diff[mask] ** 2)))
    return out


def main(
    limit: int = 0,
    models: typing.Optional[typing.Sequence[str]] = None,
) -> None:
    """
    Run the verification and write the result tables.

    Args:
        limit: Number of stations to evaluate (0 means all).
        models: Model names to evaluate (None means all).
    """
    cfg = data_2411.load_config()
    station_ids = data_2411.station_ids(cfg)
    if limit:
        station_ids = station_ids[:limit]
    slices = data_2411.year_slices(cfg)
    years = sorted(slices.keys())
    selected = tuple(models) if models else MODELS
    marker = float(cfg['obs']['calm_marker'])
    labels = time_labels(cfg)
    result_dir = pathlib.Path(cfg['paths']['result_dir'])
    result_dir.mkdir(parents=True, exist_ok=True)

    total = {name: ScoreAcc(marker) for name in selected}
    rows_year = list()
    grouped = {name: {'month': [], 'hour': [], 'lead': []} for name in selected}
    sta_ws = {name: list() for name in selected}
    sta_wd = {name: list() for name in selected}
    sector = {name: list() for name in selected}
    found = set()

    for year in years:
        ob_dir, ob_spd = load_obs(cfg, station_ids, year)
        start, end = slices[year]
        for name in selected:
            pred = load_pred(cfg, name, year)
            if pred is None:
                continue
            found.add(name)
            pr_dir = pred[..., 0]
            pr_spd = pred[..., 1]
            acc = ScoreAcc(marker)
            acc.add(ob_dir, ob_spd, pr_dir, pr_spd)
            total[name].add(ob_dir, ob_spd, pr_dir, pr_spd)
            item = acc.result()
            item['model'] = name
            item['year'] = year
            rows_year.append(item)

            for key, n_group in (('month', 12), ('hour', 24), ('lead', 24)):
                lab = labels[key][start:end, :, None]
                grouped[name][key].append(
                    group_rmse(ob_spd, pr_spd, lab, n_group)
                )
            sta_ws[name].append(
                np.sqrt(np.nanmean((pr_spd - ob_spd) ** 2, axis=(0, 1)))
            )
            delta = (
                pr_dir.astype(np.float64) - ob_dir.astype(np.float64) + 180.0
            ) % 360.0 - 180.0
            sta_wd[name].append(
                np.sqrt(np.nanmean(delta ** 2, axis=(0, 1)))
            )
            sector[name].append(
                dir_group_rmse(ob_dir, ob_spd, pr_dir, pr_spd)
            )
        print('Verified year {0}'.format(year), flush=True)

    if not found:
        print('No prediction file found', flush=True)
        return

    rows_total = list()
    for name in selected:
        if name not in found:
            continue
        item = total[name].result()
        item['model'] = name
        rows_total.append(item)

    overall = pd.DataFrame(rows_total)
    by_year = pd.DataFrame(rows_year)
    overall.to_csv(
        result_dir / 'table5_overall.csv', index=False, encoding='utf-8-sig'
    )
    by_year.to_csv(
        result_dir / 'table5_by_year.csv', index=False, encoding='utf-8-sig'
    )
    overall[overall['model'].isin(['LightGBM-A', 'LightGBM-B'])].to_csv(
        result_dir / 'table4_ab.csv', index=False, encoding='utf-8-sig'
    )

    for key in ('month', 'hour', 'lead'):
        data = {}
        for name in selected:
            if grouped[name][key]:
                data[name] = safe_nanmean(
                    np.vstack(grouped[name][key]), axis=0
                )
        if data:
            pd.DataFrame(data).to_csv(
                result_dir / 'group_{0}_rmse.csv'.format(key),
                index_label='group', encoding='utf-8-sig'
            )

    for tag, store in (('station_ws', sta_ws), ('station_wd', sta_wd)):
        data = {}
        for name in selected:
            if store[name]:
                data[name] = safe_nanmean(
                    np.vstack(store[name]), axis=0
                )
        if data:
            pd.DataFrame(data, index=station_ids).to_csv(
                result_dir / 'group_{0}_rmse.csv'.format(tag),
                index_label='station', encoding='utf-8-sig'
            )

    data = {}
    for name in selected:
        if sector[name]:
            speed_rows = [item[0] for item in sector[name]]
            dir_rows = [item[1] for item in sector[name]]
            data[name] = np.array([
                safe_nanmean(np.vstack(speed_rows), axis=0),
                safe_nanmean(np.vstack(dir_rows), axis=0),
            ], dtype=np.float64)
    if data:
        speed_tab = {name: data[name][0] for name in data}
        dir_tab = {name: data[name][1] for name in data}
        index = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        pd.DataFrame(speed_tab, index=index).to_csv(
            result_dir / 'group_sector_speed_rmse.csv',
            index_label='sector', encoding='utf-8-sig'
        )
        pd.DataFrame(dir_tab, index=index).to_csv(
            result_dir / 'group_sector_dir_rmse.csv',
            index_label='sector', encoding='utf-8-sig'
        )

    print('Result dir: {0}'.format(result_dir), flush=True)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Verify 2411-station correction models'
    )
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--models', type=str, default='')
    return parser.parse_args()


if __name__ == '__main__':
    print('Program verify_2411.py started', flush=True)
    args = parse_args()
    model_list = [
        item.strip() for item in args.models.split(',') if item.strip()
    ]
    main(limit=args.limit, models=model_list or None)




