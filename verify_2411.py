#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification for the 2411-station experiment.

Loads the per-fold predictions and the aligned observations, computes the
overall scores (CC, MAE, RMSE, MRE, OA, kappa, TS) and the grouped RMSE
statistics (year, month, lead time, hour of day, station, wind direction),
and writes the results as CSV files.

Founded in 2026-10-29
Modified in 2026-11-23
@author: yinlb
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
# Beaufort-like wind-speed thresholds (m/s).
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


def dir_grade(
    direction: np.ndarray, include_calm: bool = False
) -> np.ndarray:
    """
    Map wind direction to an 8-sector grade (0 = north); NaN becomes -1.

    Args:
        direction: Wind direction array.
        include_calm: Count the calm-wind marker as a real direction. The
            marker is taken modulo 360 before binning (999017 -> 17 deg), so
            it lands in the north sector; this reproduces the legacy
            calm-inclusive statistics. When False the marker is excluded.

    Returns:
        np.ndarray: Integer sector array.
    """
    grade = np.full(direction.shape, -1, dtype=np.int16)
    if include_calm:
        valid = ~np.isnan(direction)
    else:
        valid = (~np.isnan(direction)) & (direction != 999017)
    angle = np.where(
        valid, np.asarray(direction, dtype=np.float32) % 360.0, 0.0
    )
    grade[valid] = np.floor(
        ((angle[valid] + 22.5) % 360.0) / 45.0
    ).astype(np.int16)
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
    return counts.reshape(n_cls, n_cls).astype(np.int64)


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


def cumulative_threat_score(cm: np.ndarray) -> np.ndarray:
    """
    Compute threat scores for exceedance of each wind-speed threshold.

    Args:
        cm: Discrete speed-grade confusion matrix.

    Returns:
        np.ndarray: TS for grades above 0 through 8, equivalently speeds
        exceeding the eight thresholds in ``THRES``.
    """
    out = np.full(N_GRADE - 1, np.nan, dtype=np.float32)
    for grade in range(1, N_GRADE):
        hit = float(cm[grade:, grade:].sum())
        observed = float(cm[grade:, :].sum())
        predicted = float(cm[:, grade:].sum())
        denom = observed + predicted - hit
        if denom > 0:
            out[grade - 1] = hit / denom
    return out


def circular_delta(
    ob_dir: np.ndarray, pr_dir: np.ndarray, marker: float = 999017.0
) -> np.ndarray:
    """
    Return circular direction errors, excluding calm markers and NaNs.

    Args:
        ob_dir: Observed directions in degrees.
        pr_dir: Predicted directions in degrees.
        marker: Calm-wind direction marker.

    Returns:
        np.ndarray: Signed errors in [-180, 180), with NaN for invalid pairs.
    """
    ob = np.asarray(ob_dir, dtype=np.float32)
    pred = np.asarray(pr_dir, dtype=np.float32)
    valid = (
        np.isfinite(ob) & np.isfinite(pred)
        & (ob != marker) & (pred != marker)
    )
    delta = np.full(ob.shape, np.nan, dtype=np.float32)
    delta[valid] = (pred[valid] - ob[valid] + 180.0) % 360.0 - 180.0
    return delta


def common_masks(
    ob_dir: np.ndarray,
    ob_spd: np.ndarray,
    predictions: typing.Sequence[np.ndarray],
    marker: float,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Build common wind-speed and non-calm wind-direction evaluation masks.

    Args:
        ob_dir: Observed directions.
        ob_spd: Observed speeds.
        predictions: Model arrays with (direction, speed) in the last axis.
        marker: Calm-wind direction marker.

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: Common speed and direction masks.
        The direction mask additionally requires observed and predicted speeds.
    """
    mask_ws = np.isfinite(ob_spd)
    mask_wd = (
        mask_ws & np.isfinite(ob_dir) & (ob_dir != marker)
    )
    for pred in predictions:
        pr_dir = pred[..., 0]
        pr_spd = pred[..., 1]
        mask_ws = mask_ws & np.isfinite(pr_spd)
        mask_wd = (
            mask_wd & np.isfinite(pr_dir) & np.isfinite(pr_spd)
            & (pr_dir != marker)
        )
    return mask_ws, mask_wd


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
    obs = ob_spd[mask].astype(np.float32)
    pred = pr_spd[mask].astype(np.float32)
    out['ws_cc'] = float(np.corrcoef(obs, pred)[0, 1])
    out['ws_mae'] = float(np.mean(np.abs(pred - obs)))
    out['ws_rmse'] = float(np.sqrt(np.mean((pred - obs) ** 2)))
    denom = pred + obs
    keep = denom > 0
    ratio = np.abs((pred[keep] - obs[keep]) / denom[keep])
    out['ws_mre'] = float(np.mean(ratio))

    cm_ws = confusion(speed_grade(obs), speed_grade(pred), N_GRADE)
    out['ws_oa'], out['ws_kappa'] = oa_kappa(cm_ws)
    ts_val = cumulative_threat_score(cm_ws)
    for grade in range(1, N_GRADE):
        out['ws_ts{0}'.format(grade)] = float(ts_val[grade - 1])

    delta = circular_delta(ob_dir, pr_dir)
    delta = delta[np.isfinite(delta)]
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
    n_hour = int(data_2411.target_hours(cfg).size)
    direction = np.full((n_sample, n_hour, n_station), np.nan, np.float32)
    speed = np.full((n_sample, n_hour, n_station), np.nan, np.float32)
    for index, sta in enumerate(station_ids):
        arrays = data_2411.load_arrays(sta, cfg)
        if arrays is None:
            continue
        direction[:, :, index] = arrays['dir'][start:end]
        speed[:, :, index] = arrays['spd'][start:end]
    return direction, speed


def pred_path(cfg: dict, model: str, year: int) -> pathlib.Path:
    """
    Return the prediction file path of one model and one fold.

    Args:
        cfg: Configuration dictionary.
        model: Model name.
        year: Held-out year.

    Returns:
        pathlib.Path: Prediction file path.
    """
    return (
        data_2411.pred_root(cfg)
        / '{0}_{1}.npy'.format(FILE_NAMES[model], year)
    )


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
    path = pred_path(cfg, model, year)
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
    n_hour = int(data_2411.target_hours(cfg).size)
    # months arrive as 1-12, but group_rmse expects zero-based group ids.
    month = times.month.to_numpy().reshape(n_sample, n_hour) - 1
    hour = times.hour.to_numpy().reshape(n_sample, n_hour)
    lead = np.broadcast_to(np.arange(n_hour), (n_sample, n_hour)).copy()
    return {'month': month, 'hour': hour, 'lead': lead}


class ScoreAcc(object):
    """Streaming accumulator with optional common-sample masks."""

    def __init__(
        self, marker: float = 999017.0, calm_mode: str = 'exclude'
    ) -> None:
        """
        Initialize the accumulator.

        Args:
            marker: Calm-wind direction marker.
            calm_mode: 'exclude' drops the marker from the direction grades;
                'include' keeps it for the legacy own-sample statistics.
        """
        self.marker = float(marker)
        self.include_calm = calm_mode == 'include'
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
        self.cm_ws = np.zeros((N_GRADE, N_GRADE), dtype=np.int64)
        self.n_wd = 0
        self.n_wd_base = 0
        self.s_wabs = 0.0
        self.s_wsq = 0.0
        self.cm_wd = np.zeros((N_DIR, N_DIR), dtype=np.int64)
        self.cm_wd_grade = np.zeros(
            (N_GRADE, N_DIR, N_DIR), dtype=np.int64
        )

    def add(
        self, ob_dir: np.ndarray, ob_spd: np.ndarray,
        pr_dir: np.ndarray, pr_spd: np.ndarray,
        mask_ws: typing.Optional[np.ndarray] = None,
        mask_wd: typing.Optional[np.ndarray] = None,
    ) -> None:
        """
        Add one block of observations and predictions.

        Args:
            ob_dir: Observed direction.
            ob_spd: Observed speed.
            pr_dir: Predicted direction.
            pr_spd: Predicted speed.
            mask_ws: Optional common wind-speed mask.
            mask_wd: Optional common non-calm wind-direction mask.
        """
        eligible_wd = (
            np.isfinite(ob_dir) & np.isfinite(ob_spd)
            & (ob_dir != self.marker)
        )
        self.n_wd_base += int(eligible_wd.sum())

        mask = np.isfinite(ob_spd) & np.isfinite(pr_spd)
        if mask_ws is not None:
            mask = mask & mask_ws
        obs = ob_spd[mask].astype(np.float32)
        pred = pr_spd[mask].astype(np.float32)
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
            eligible_wd & np.isfinite(pr_dir) & (pr_dir != self.marker)
        )
        if mask_wd is not None:
            mask_dir = mask_dir & mask_wd
        delta = circular_delta(ob_dir, pr_dir, self.marker)
        delta = delta[mask_dir]
        self.n_wd += int(delta.size)
        self.s_wabs += float(np.sum(np.abs(delta)))
        self.s_wsq += float(np.sum(delta ** 2))

        mask_cm = np.isfinite(ob_dir) & np.isfinite(pr_dir)
        if not self.include_calm or mask_wd is not None:
            mask_cm = (
                mask_cm & (ob_dir != self.marker) & (pr_dir != self.marker)
            )
        if mask_wd is not None:
            mask_cm = mask_cm & mask_wd
        self.cm_wd += confusion(
            dir_grade(np.where(mask_cm, ob_dir, np.nan), self.include_calm),
            dir_grade(np.where(mask_cm, pr_dir, np.nan), self.include_calm),
            N_DIR,
        )
        ob_grade = speed_grade(ob_spd)
        for grade in range(N_GRADE):
            in_grade = mask_cm & (ob_grade == grade)
            if in_grade.any():
                self.cm_wd_grade[grade] += confusion(
                    dir_grade(ob_dir[in_grade], self.include_calm),
                    dir_grade(pr_dir[in_grade], self.include_calm), N_DIR,
                )

    def merge(self, other: 'ScoreAcc') -> None:
        """
        Merge another accumulator without rescanning its samples.

        Args:
            other: Accumulator with the same marker and calm mode.

        Raises:
            ValueError: If the accumulator settings differ.
        """
        if (
            self.marker != other.marker
            or self.include_calm != other.include_calm
        ):
            raise ValueError(
                'Cannot merge accumulators with different settings'
            )
        scalar_names = (
            'n_ws', 's_abs', 's_sq', 's_mre', 'n_mre', 's_o', 's_p',
            's_oo', 's_pp', 's_op', 'n_wd', 'n_wd_base', 's_wabs', 's_wsq',
        )
        for name in scalar_names:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.cm_ws += other.cm_ws
        self.cm_wd += other.cm_wd
        self.cm_wd_grade += other.cm_wd_grade

    def result(self) -> typing.Dict[str, float]:
        """
        Compute the accumulated scores.

        Returns:
            typing.Dict[str, float]: Score dictionary, including sample counts,
            wind-speed bias and direction coverage.
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
        out['ws_bias'] = (self.s_p - self.s_o) / n_ws
        out['ws_mre'] = (
            self.s_mre / self.n_mre if self.n_mre > 0 else np.nan
        )
        out['ws_oa'], out['ws_kappa'] = oa_kappa(self.cm_ws)
        ts_val = cumulative_threat_score(self.cm_ws)
        for grade in range(1, N_GRADE):
            out['ws_ts{0}'.format(grade)] = float(ts_val[grade - 1])
        n_wd = max(self.n_wd, 1)
        out['wd_mae'] = self.s_wabs / n_wd
        out['wd_rmse'] = float(np.sqrt(self.s_wsq / n_wd))
        out['wd_oa'], out['wd_kappa'] = oa_kappa(self.cm_wd)
        out['n_ws'] = self.n_ws
        out['n_wd'] = self.n_wd
        out['n_wd_base'] = self.n_wd_base
        out['wd_coverage'] = (
            self.n_wd / self.n_wd_base if self.n_wd_base > 0 else np.nan
        )
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
    out = np.full((2, N_DIR), np.nan, dtype=np.float32)
    diff_spd = pr_spd - ob_spd
    delta = (
        pr_dir.astype(np.float32) - ob_dir.astype(np.float32) + 180.0
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


def group_error_stats(
    delta: np.ndarray, label: np.ndarray, n_group: int
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Accumulate squared-error sums and sample counts for each group.

    Args:
        delta: Signed errors, with NaN for excluded samples.
        label: Group labels, broadcastable to ``delta``.
        n_group: Number of groups.

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: Squared-error sums and counts.
    """
    delta = np.asarray(delta, dtype=np.float32)
    lab = label
    while lab.ndim < delta.ndim:
        lab = lab[..., None]
    lab = np.broadcast_to(lab, delta.shape)
    valid = np.isfinite(delta) & (lab >= 0) & (lab < n_group)
    group_id = lab[valid].astype(np.int64)
    values = delta[valid]
    count = np.bincount(group_id, minlength=n_group).astype(np.int64)
    sse = np.bincount(
        group_id, weights=values ** 2, minlength=n_group
    ).astype(np.float64)
    return sse, count


def rmse_from_stats(sse: np.ndarray, count: np.ndarray) -> np.ndarray:
    """
    Convert pooled squared-error sums and counts to RMSE.

    Args:
        sse: Squared-error sums.
        count: Sample counts, with zero for an empty group.

    Returns:
        np.ndarray: RMSE values, with NaN for empty groups.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(count > 0, np.sqrt(sse / np.maximum(count, 1)), np.nan)


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
    sse, count = group_error_stats(pred - obs, label, n_group)
    return rmse_from_stats(sse, count)


def group_dir_rmse(
    ob_dir: np.ndarray, pr_dir: np.ndarray, label: np.ndarray,
    n_group: int, marker: float = 999017.0,
) -> np.ndarray:
    """
    Compute the circular direction RMSE for every group of a label array.

    Args:
        ob_dir: Observed direction.
        pr_dir: Predicted direction.
        label: Group labels, either the same shape as the data or
            broadcastable to it.
        n_group: Number of groups.
        marker: Calm-wind direction marker.

    Returns:
        np.ndarray: Direction RMSE per group.
    """
    delta = circular_delta(ob_dir, pr_dir, marker)
    return group_rmse(np.zeros_like(delta), delta, label, n_group)


def write_text(
    frame: pd.DataFrame,
    table_dir: pathlib.Path,
    name: str,
    title: str,
    index: bool = False,
) -> None:
    """
    Print a result table and save the identical text as a TXT file.

    Args:
        frame: Table to report.
        table_dir: Directory holding the result tables.
        name: File stem of the TXT file.
        title: Heading printed above the rows.
        index: Whether to include the frame index.
    """
    text = frame.to_string(index=index)
    lines = [title, '=' * len(title), text, '']
    path = table_dir / '{0}.txt'.format(name)
    with open(path, 'w', encoding='utf-8') as file_handle:
        file_handle.write('\n'.join(lines))
    print(title, flush=True)
    print(text, flush=True)


def main(
    limit: int = 0,
    models: typing.Optional[typing.Sequence[str]] = None,
    calm_mode: str = 'exclude',
) -> None:
    """
    Run own-sample and common-sample verification and write result tables.

    Args:
        limit: Number of stations to evaluate (0 means all).
        models: Model names to evaluate (None means all).
        calm_mode: 'exclude' drops calm samples from the direction grades;
            'include' affects only the legacy own-sample tables.
    """
    cfg = data_2411.load_config()
    station_ids = data_2411.station_ids(cfg)
    if limit:
        station_ids = station_ids[:limit]
    slices = data_2411.year_slices(cfg)
    years = sorted(slices.keys())
    selected = tuple(models) if models else MODELS
    unknown = [name for name in selected if name not in MODELS]
    if unknown:
        print(
            'Unknown model: {0}; valid: {1}'.format(
                ', '.join(unknown), ', '.join(MODELS)
            ),
            flush=True,
        )
        return
    marker = float(cfg['obs']['calm_marker'])
    labels = time_labels(cfg)
    years = [
        year for year in years
        if any(pred_path(cfg, name, year).exists() for name in selected)
    ]
    if not years:
        print('No prediction file found', flush=True)
        return
    common_years = [
        year for year in years
        if all(pred_path(cfg, name, year).exists() for name in selected)
    ]
    common_mode = bool(common_years)
    print(
        'Evaluating folds: {0}; common folds: {1}'.format(
            ', '.join(str(y) for y in years),
            ', '.join(str(y) for y in common_years),
        ),
        flush=True,
    )
    result_dir = data_2411.result_root(cfg)
    table_dir = result_dir / 'tables'
    table_dir.mkdir(parents=True, exist_ok=True)

    total_own = {
        name: ScoreAcc(marker, calm_mode) for name in selected
    }
    total_common = {
        name: ScoreAcc(marker, 'exclude') for name in selected
    }
    rows_year_own = list()
    rows_year_common = list()
    n_hour = int(data_2411.target_hours(cfg).size)
    grouped = {
        name: {
            key: [np.zeros(n), np.zeros(n, dtype=np.int64)]
            for key, n in (('month', 12), ('hour', 24), ('lead', n_hour))
        }
        for name in selected
    }
    grouped_dir = {
        name: {
            key: [np.zeros(n), np.zeros(n, dtype=np.int64)]
            for key, n in (('month', 12), ('hour', 24), ('lead', n_hour))
        }
        for name in selected
    }
    station = {
        name: [
            np.zeros(len(station_ids)), np.zeros(len(station_ids), np.int64),
            np.zeros(len(station_ids)), np.zeros(len(station_ids), np.int64),
        ]
        for name in selected
    }
    sector = {
        name: [
            np.zeros((2, N_DIR)), np.zeros((2, N_DIR), dtype=np.int64)
        ]
        for name in selected
    }
    found = set()

    for year in years:
        ob_dir, ob_spd = load_obs(cfg, station_ids, year)
        start, end = slices[year]
        predictions = {
            name: load_pred(cfg, name, year) for name in selected
        }
        use_common = common_mode and year in common_years
        if use_common:
            mask_ws_common, mask_wd_common = common_masks(
                ob_dir, ob_spd, list(predictions.values()), marker
            )
        for name in selected:
            pred = predictions[name]
            if pred is None:
                continue
            found.add(name)
            pr_dir = pred[..., 0]
            pr_spd = pred[..., 1]
            acc_own = ScoreAcc(marker, calm_mode)
            acc_own.add(ob_dir, ob_spd, pr_dir, pr_spd)
            total_own[name].merge(acc_own)
            item = acc_own.result()
            item['model'] = name
            item['year'] = year
            rows_year_own.append(item)

            if use_common:
                acc = ScoreAcc(marker, 'exclude')
                acc.add(
                    ob_dir, ob_spd, pr_dir, pr_spd,
                    mask_ws=mask_ws_common, mask_wd=mask_wd_common,
                )
                total_common[name].merge(acc)
                item = acc.result()
                item['model'] = name
                item['year'] = year
                rows_year_common.append(item)
                mask_ws = mask_ws_common
                mask_wd = mask_wd_common
            elif not common_mode:
                mask_ws = np.isfinite(ob_spd) & np.isfinite(pr_spd)
                mask_wd = None
            else:
                continue

            error_ws = np.where(mask_ws, pr_spd - ob_spd, np.nan)
            delta_wd = circular_delta(ob_dir, pr_dir, marker)
            if mask_wd is not None:
                delta_wd = np.where(mask_wd, delta_wd, np.nan)
            for key, n_group in (('month', 12), ('hour', 24), ('lead', n_hour)):
                lab = labels[key][start:end, :, None]
                sse, count = group_error_stats(error_ws, lab, n_group)
                grouped[name][key][0] += sse
                grouped[name][key][1] += count
                sse, count = group_error_stats(delta_wd, lab, n_group)
                grouped_dir[name][key][0] += sse
                grouped_dir[name][key][1] += count

            station[name][0] += np.nansum(error_ws ** 2, axis=(0, 1))
            station[name][1] += np.isfinite(error_ws).sum(axis=(0, 1))
            station[name][2] += np.nansum(delta_wd ** 2, axis=(0, 1))
            station[name][3] += np.isfinite(delta_wd).sum(axis=(0, 1))

            sectors = dir_grade(ob_dir)
            sse, count = group_error_stats(error_ws, sectors, N_DIR)
            sector[name][0][0] += sse
            sector[name][1][0] += count
            sse, count = group_error_stats(delta_wd, sectors, N_DIR)
            sector[name][0][1] += sse
            sector[name][1][1] += count
        print('Verified year {0}'.format(year), flush=True)

    if not found:
        print('No prediction file found', flush=True)
        return

    rows_own = list()
    for name in selected:
        if name in found:
            item = total_own[name].result()
            item['model'] = name
            rows_own.append(item)
    own = pd.DataFrame(rows_own)
    own.to_csv(
        table_dir / 'table5_own_samples.csv', index=False,
        encoding='utf-8-sig'
    )
    own_ab = own[own['model'].isin(['LightGBM-A', 'LightGBM-B'])]
    own_ab.to_csv(
        table_dir / 'table4_ab_own_samples.csv', index=False,
        encoding='utf-8-sig'
    )
    write_text(
        own, table_dir, 'table5_own_samples', 'Table 5 own-sample appendix'
    )
    write_text(
        own_ab, table_dir, 'table4_ab_own_samples',
        'Table 4 own-sample appendix'
    )

    rows_common = list()
    for name in selected:
        if total_common[name].n_ws:
            item = total_common[name].result()
            item['model'] = name
            rows_common.append(item)
    overall = pd.DataFrame(rows_common) if common_mode else own
    by_year = pd.DataFrame(
        rows_year_common if common_mode else rows_year_own
    )
    overall.to_csv(
        table_dir / 'table5_overall.csv', index=False,
        encoding='utf-8-sig'
    )
    by_year.to_csv(
        table_dir / 'table5_by_year.csv', index=False,
        encoding='utf-8-sig'
    )
    table4 = overall[
        overall['model'].isin(['LightGBM-A', 'LightGBM-B'])
    ]
    table4.to_csv(
        table_dir / 'table4_ab.csv', index=False, encoding='utf-8-sig'
    )
    mode_title = 'common samples' if common_mode else 'own samples'
    write_text(overall, table_dir, 'table5_overall',
               'Table 5 overall ({0})'.format(mode_title))
    write_text(table4, table_dir, 'table4_ab',
               'Table 4 angular vs vector ({0})'.format(mode_title))
    write_text(by_year, table_dir, 'table5_by_year',
               'Table 5 by year ({0})'.format(mode_title))

    grade_data = {}
    count_data = {}
    grade_source = total_common if common_mode else total_own
    for name in overall['model']:
        grade_data[name] = [
            oa_kappa(matrix)[1]
            for matrix in grade_source[name].cm_wd_grade[1:]
        ]
        count_data[name] = [
            float(matrix.sum()) for matrix in grade_source[name].cm_wd_grade[1:]
        ]
    if grade_data:
        index = range(1, N_GRADE)
        kappa_tab = pd.DataFrame(grade_data, index=index)
        count_tab = pd.DataFrame(count_data, index=index)
        kappa_tab.to_csv(
            table_dir / 'group_grade_wd_kappa.csv',
            index_label='grade', encoding='utf-8-sig'
        )
        count_tab.to_csv(
            table_dir / 'group_grade_wd_count.csv',
            index_label='grade', encoding='utf-8-sig'
        )
        write_text(
            kappa_tab, table_dir, 'group_grade_wd_kappa',
            'Direction kappa by observed speed grade', index=True,
        )
        write_text(
            count_tab, table_dir, 'group_grade_wd_count',
            'Common direction samples by observed speed grade', index=True,
        )

    for key in ('month', 'hour', 'lead'):
        data = {}
        dir_data = {}
        for name in overall['model']:
            sse, count = grouped[name][key]
            data[name] = rmse_from_stats(sse, count)
            sse, count = grouped_dir[name][key]
            dir_data[name] = rmse_from_stats(sse, count)
        if data:
            grid = pd.DataFrame(data)
            grid.to_csv(
                table_dir / 'group_{0}_rmse.csv'.format(key),
                index_label='group', encoding='utf-8-sig'
            )
            write_text(
                grid, table_dir, 'group_{0}_rmse'.format(key),
                'Wind-speed RMSE by {0}'.format(key), index=True,
            )
        if dir_data:
            grid = pd.DataFrame(dir_data)
            grid.to_csv(
                table_dir / 'group_{0}_dir_rmse.csv'.format(key),
                index_label='group', encoding='utf-8-sig'
            )
            write_text(
                grid, table_dir, 'group_{0}_dir_rmse'.format(key),
                'Wind-direction RMSE by {0}'.format(key), index=True,
            )

    station_values = {}
    for name in overall['model']:
        ws_sse, ws_count, wd_sse, wd_count = station[name]
        station_values[name] = (
            rmse_from_stats(ws_sse, ws_count),
            rmse_from_stats(wd_sse, wd_count),
        )
    if station_values:
        pd.DataFrame(
            {name: value[0] for name, value in station_values.items()},
            index=station_ids,
        ).to_csv(
            table_dir / 'group_station_ws_rmse.csv',
            index_label='station', encoding='utf-8-sig'
        )
        pd.DataFrame(
            {name: value[1] for name, value in station_values.items()},
            index=station_ids,
        ).to_csv(
            table_dir / 'group_station_wd_rmse.csv',
            index_label='station', encoding='utf-8-sig'
        )

    data = {}
    for name in overall['model']:
        sse, count = sector[name]
        data[name] = rmse_from_stats(sse, count)
    if data:
        sector_index = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        speed_tab = pd.DataFrame(
            {name: data[name][0] for name in data}, index=sector_index
        )
        dir_tab = pd.DataFrame(
            {name: data[name][1] for name in data}, index=sector_index
        )
        speed_tab.to_csv(
            table_dir / 'group_sector_speed_rmse.csv',
            index_label='sector', encoding='utf-8-sig'
        )
        dir_tab.to_csv(
            table_dir / 'group_sector_dir_rmse.csv',
            index_label='sector', encoding='utf-8-sig'
        )
        write_text(
            speed_tab, table_dir, 'group_sector_speed_rmse',
            'Wind-speed RMSE by observed sector', index=True,
        )
        write_text(
            dir_tab, table_dir, 'group_sector_dir_rmse',
            'Wind-direction RMSE by observed sector', index=True,
        )

    summary = list()
    summary.append('2411-station verification summary')
    summary.append('=' * 32)
    summary.append(
        'stations {0}  folds {1}  common_folds {2}  mode {3}'.format(
            len(station_ids), len(years), len(common_years), mode_title
        )
    )
    summary.append('models {0}'.format(', '.join(overall['model'])))
    summary.append('')
    summary.append('Table 5 overall')
    summary.append(overall.to_string(index=False))
    summary.append('')
    summary.append('Table 4 angular vs vector')
    summary.append(table4.to_string(index=False))

    base = overall.set_index('model')
    if 'ECMWF-IFS' in base.index:
        ref = base.loc['ECMWF-IFS']
        summary.append('')
        summary.append('Change over ECMWF-IFS (%)')
        for name in base.index:
            if name == 'ECMWF-IFS':
                continue
            row = base.loc[name]
            summary.append(
                '  {0:12s} ws_rmse {1:+7.2f}  wd_rmse {2:+7.2f}  '
                'cc {3:+7.2f}  kappa {4:+7.2f}'.format(
                    name,
                    100.0 * (ref['ws_rmse'] - row['ws_rmse'])
                    / ref['ws_rmse'],
                    100.0 * (ref['wd_rmse'] - row['wd_rmse'])
                    / ref['wd_rmse'],
                    100.0 * (row['ws_cc'] - ref['ws_cc']) / ref['ws_cc'],
                    100.0 * (row['wd_kappa'] - ref['wd_kappa'])
                    / ref['wd_kappa'],
                )
            )

    summary.append('')
    summary.append('Station RMSE distribution (mean / median / p10 / p90)')
    for name, values in station_values.items():
        ws_val, wd_val = values
        summary.append(
            '  {0:12s} ws {1:6.3f} {2:6.3f} {3:6.3f} {4:6.3f}   '
            'wd {5:6.2f} {6:6.2f} {7:6.2f} {8:6.2f}'.format(
                name,
                np.nanmean(ws_val), np.nanmedian(ws_val),
                np.nanpercentile(ws_val, 10), np.nanpercentile(ws_val, 90),
                np.nanmean(wd_val), np.nanmedian(wd_val),
                np.nanpercentile(wd_val, 10), np.nanpercentile(wd_val, 90),
            )
        )

    summary_text = '\n'.join(summary)
    with open(
        table_dir / 'summary.txt', 'w', encoding='utf-8'
    ) as file_handle:
        file_handle.write(summary_text + '\n')
    print(summary_text, flush=True)
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
    parser.add_argument(
        '--calm-mode', type=str, default='exclude',
        choices=('exclude', 'include'),
        help='exclude calm samples from the direction grades (default) or '
             'count them as in the legacy calm-inclusive statistics',
    )
    return parser.parse_args()


if __name__ == '__main__':
    print('Program verify_2411.py started', flush=True)
    args = parse_args()
    model_list = [
        item.strip() for item in args.models.split(',') if item.strip()
    ]
    main(
        limit=args.limit,
        models=model_list or None,
        calm_mode=args.calm_mode,
    )




