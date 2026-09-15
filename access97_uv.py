#!user/bin.python3
# -*- coding: utf-8 -*-

"""
Evaluation metrics for 97-station UV-component wind forecasts.

This module implements the Acc class for wind-direction and wind-speed
verification scores, and a main routine that loads model outputs, computes
scores, and draws comparison charts (bars, violin plots, radar plots).

Founded in 2024-10-11
Modified in 2026-09-15
@author: yinlb
"""

import os
import copy
import pathlib
import sys
import typing

import arrow
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from meteva import base as meb

from evaluation import access97_uv_plots
from src import utils


THRES = (0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2)
ELEMENT_NAME_LIST = [
    'T_2m', 'MSL', 'D_2m', 'U_10m', 'V_10m', 'T_1000hPa',
    'GH_1000hPa', 'R_1000hPa', 'U_1000hPa', 'V_1000hPa',
    'T_925hPa', 'GH_925hPa', 'R_925hPa', 'U_925hPa', 'V_925hPa',
    'T_850hPa', 'GH_850hPa', 'R_850hPa', 'U_850hPa', 'V_850hPa',
    'T_700hPa', 'GH_700hPa', 'R_700hPa', 'U_700hPa', 'V_700hPa',
    'T_500hPa', 'GH_500hPa', 'R_500hPa', 'U_500hPa', 'V_500hPa'
]


class Acc:
    """
    Wind-direction and wind-speed verification metrics.

    Attributes:
        thres (tuple): Wind-speed thresholds for Beaufort-like grades.
        n_grades (int): Number of wind-speed grades, equals len(thres) + 1.
        ob_wd (np.ndarray): Observed wind direction.
        pr_wd (np.ndarray): Predicted wind direction.
        ob_wd_grade (np.ndarray): Observed wind-direction grade (0-7).
        pr_wd_grade (np.ndarray): Predicted wind-direction grade (0-7).
        wd_cm (np.ndarray): Confusion matrix for wind-direction grades.
        wd_n (int): Total number of wind-direction samples.
        ob_ws (np.ndarray): Observed wind speed.
        pr_ws (np.ndarray): Predicted wind speed.
        ob_ws_grade (np.ndarray): Observed wind-speed grade.
        pr_ws_grade (np.ndarray): Predicted wind-speed grade.
        ws_cm (np.ndarray): Confusion matrix for wind-speed grades.
        ws_n (int): Total number of wind-speed samples.
    """

    def __init__(self, ob: np.ndarray, pr: np.ndarray) -> None:
        """
        Initialize verification object from observation and prediction.

        Args:
            ob: Observation array with wind direction in the first channel
                and wind speed in the second channel.
            pr: Prediction array with the same shape as ob.
        """
        self.thres = THRES
        self.n_grades = len(self.thres) + 1

        self.ob_wd = ob[..., 0]
        self.pr_wd = pr[..., 0]

        self.ob_wd_grade = np.zeros_like(self.ob_wd, dtype=np.int_) - 1
        for i in range(8):
            if i == 0:
                mask = (self.ob_wd < 22.5) | (self.ob_wd >= 337.5)
                self.ob_wd_grade[mask] = i
            else:
                mask = (
                    (self.ob_wd >= 45 * i - 22.5)
                    & (self.ob_wd < 45 * i + 22.5)
                )
                self.ob_wd_grade[mask] = i

        self.pr_wd_grade = np.zeros_like(self.pr_wd, dtype=np.int_) - 1
        for i in range(8):
            if i == 0:
                mask = (
                    ((self.pr_wd < 22.5) & (self.pr_wd >= 0))
                    | ((self.pr_wd >= 337.5) & (self.pr_wd <= 360))
                )
                self.pr_wd_grade[mask] = i
            else:
                mask = (
                    (self.pr_wd >= 45 * i - 22.5)
                    & (self.pr_wd < 45 * i + 22.5)
                )
                self.pr_wd_grade[mask] = i

        self.wd_cm = np.zeros((8, 8), dtype=np.int_)
        for i in range(8):
            for j in range(8):
                self.wd_cm[i, j] = np.sum(
                    (self.ob_wd_grade == i) & (self.pr_wd_grade == j)
                )
        self.wd_n = np.sum(self.wd_cm)

        self.ob_ws = ob[..., 1]
        self.pr_ws = pr[..., 1]

        self.ob_ws_grade = np.zeros_like(self.ob_ws, dtype=np.int_) - 1
        self.ob_ws_grade[~np.isnan(self.ob_ws)] = 0
        for i in range(self.n_grades - 1):
            self.ob_ws_grade[self.ob_ws >= self.thres[i]] = i + 1

        self.pr_ws_grade = np.zeros_like(self.pr_ws, dtype=np.int_) - 1
        self.pr_ws_grade[~np.isnan(self.pr_ws)] = 0
        for i in range(self.n_grades - 1):
            self.pr_ws_grade[self.pr_ws >= self.thres[i]] = i + 1

        self.ws_cm = np.zeros(
            (self.n_grades + 1, self.n_grades + 1), dtype=np.int_
        )
        for i in range(self.n_grades + 1):
            for j in range(self.n_grades + 1):
                self.ws_cm[i, j] = np.sum(
                    (self.ob_ws_grade == i) & (self.pr_ws_grade == j)
                )
        self.ws_n = np.sum(self.ws_cm)

    def _get_wd_delta(
        self, wd_grade: typing.Optional[int] = None
   ) -> np.ndarray:
        """
        Compute wind-direction error array after circular adjustment.

        Args:
            wd_grade: Wind-direction grade (0-7) to filter observations, or
                None to use all samples.

        Returns:
            Wind-direction error array in degrees, range [-180, 180].

        Raises:
            AssertionError: If wd_grade is not None or an integer 0-7.
        """
        assert wd_grade in (None, 0, 1, 2, 3, 4, 5, 6, 7)
        mask = (~np.isnan(self.ob_wd)) & (~np.isnan(self.pr_wd))
        if wd_grade is not None:
            mask &= self.ob_wd_grade == wd_grade
        ob_wd = self.ob_wd[mask]
        pr_wd = self.pr_wd[mask]
        delta = pr_wd - ob_wd
        delta = delta[ob_wd != 999017]
        delta[delta > 180] -= 360
        delta[delta < -180] += 360
        return delta

    def get_wd_me(
        self, wd_grade: typing.Optional[int] = None
   ) -> float:
        """
        Compute mean wind-direction error.

        Args:
            wd_grade: Wind-direction grade (0-7) to filter observations, or
                None to use all samples.

        Returns:
            Mean wind-direction error in degrees.

        Raises:
            AssertionError: If wd_grade is not None or an integer 0-7.
        """
        delta = self._get_wd_delta(wd_grade)
        return float(np.mean(delta))

    def get_wd_mae(
        self, wd_grade: typing.Optional[int] = None
   ) -> float:
        """
        Compute mean absolute wind-direction error.

        Args:
            wd_grade: Wind-direction grade (0-7) to filter observations, or
                None to use all samples.

        Returns:
            Mean absolute wind-direction error in degrees.

        Raises:
            AssertionError: If wd_grade is not None or an integer 0-7.
        """
        delta = self._get_wd_delta(wd_grade)
        return float(np.mean(np.abs(delta)))

    def get_wd_rmse(
        self, wd_grade: typing.Optional[int] = None
   ) -> float:
        """
        Compute root-mean-square wind-direction error.

        Args:
            wd_grade: Wind-direction grade (0-7) to filter observations, or
                None to use all samples.

        Returns:
            Root-mean-square wind-direction error in degrees.

        Raises:
            AssertionError: If wd_grade is not None or an integer 0-7.
        """
        delta = self._get_wd_delta(wd_grade)
        return float(np.mean(delta ** 2) ** 0.5)

    def get_wd_cs(
        self, wd_grade: typing.Optional[int] = None
   ) -> float:
        """
        Compute mean cosine of wind-direction error.

        Args:
            wd_grade: Wind-direction grade (0-7) to filter observations, or
                None to use all samples.

        Returns:
            Mean cosine of wind-direction error.

        Raises:
            AssertionError: If wd_grade is not None or an integer 0-7.
        """
        delta = self._get_wd_delta(wd_grade)
        return float(np.mean(np.cos(np.deg2rad(delta))))

    def get_wd_cm(self) -> np.ndarray:
        """
        Return the wind-direction confusion matrix.

        Returns:
            Wind-direction confusion matrix with shape (8, 8).
        """
        return self.wd_cm

    def get_wd_oa(
        self, is_ws: bool = False
   ) -> typing.Union[float, np.ndarray]:
        """
        Compute overall accuracy (OA) for wind-direction grades.

        Args:
            is_ws: Whether to compute OA per wind-speed grade.

        Returns:
            Scalar OA when is_ws is False, otherwise an array of OA values.
        """
        if is_ws:
            oa = np.zeros(self.n_grades - 1, dtype=np.float32) + np.nan
            for i in range(1, self.n_grades):
                wd_cm = np.zeros((8, 8), dtype=np.int_)
                for j in range(8):
                    for k in range(8):
                        mask = (
                            (self.ob_ws_grade == i)
                            & (self.ob_wd_grade == j)
                            & (self.pr_wd_grade == k)
                        )
                        wd_cm[j, k] = np.sum(mask)
                oa[i - 1] = np.sum(np.diag(wd_cm)) / np.sum(wd_cm)
        else:
            oa = np.sum(np.diag(self.wd_cm)) / self.wd_n
        return oa

    def get_wd_kappa(
        self, is_ws: bool = False
   ) -> typing.Union[float, np.ndarray]:
        """
        Compute Cohen's kappa for wind-direction grades.

        Args:
            is_ws: Whether to compute kappa per wind-speed grade.

        Returns:
            Scalar kappa when is_ws is False, otherwise an array of kappa.
        """
        if is_ws:
            kappa0 = np.zeros(self.n_grades - 1, dtype=np.float32) + np.nan
            for i in range(1, self.n_grades):
                wd_cm = np.zeros((8, 8), dtype=np.int_)
                for j in range(8):
                    for k in range(8):
                        mask = (
                            (self.ob_ws_grade == i)
                            & (self.ob_wd_grade == j)
                            & (self.pr_wd_grade == k)
                        )
                        wd_cm[j, k] = np.sum(mask)
                row_total = np.sum(wd_cm, axis=1).astype(np.float32)
                col_total = np.sum(wd_cm, axis=0).astype(np.float32)
                total = np.sum(wd_cm)
                pe = np.sum(row_total * col_total) / total / total
                oa = np.sum(np.diag(wd_cm)) / total
                kappa0[i - 1] = (oa - pe) / (1 - pe)
        else:
            row_total = np.sum(self.wd_cm, axis=1).astype(np.float32)
            col_total = np.sum(self.wd_cm, axis=0).astype(np.float32)
            pe = np.sum(row_total * col_total) / self.wd_n / self.wd_n
            kappa0 = (self.get_wd_oa() - pe) / (1 - pe)
        return kappa0

    def get_ws_me(self) -> float:
        """
        Compute mean wind-speed error.

        Returns:
            Mean wind-speed error (prediction minus observation) in m/s.
        """
        mask = (~np.isnan(self.ob_ws)) & (~np.isnan(self.pr_ws))
        ob_ws = self.ob_ws[mask]
        pr_ws = self.pr_ws[mask]
        return float(np.mean(pr_ws - ob_ws))

    def get_ws_mae(
        self, wd_grade: typing.Optional[int] = None
   ) -> float:
        """
        Compute mean absolute wind-speed error.

        Args:
            wd_grade: Wind-direction grade (0-7) to filter observations, or
                None to use all samples.

        Returns:
            Mean absolute wind-speed error in m/s.

        Raises:
            AssertionError: If wd_grade is not None or an integer 0-7.
        """
        assert wd_grade in (None, 0, 1, 2, 3, 4, 5, 6, 7)
        mask = (~np.isnan(self.ob_ws)) & (~np.isnan(self.pr_ws))
        if wd_grade is not None:
            mask &= self.ob_wd_grade == wd_grade
        ob_ws = self.ob_ws[mask]
        pr_ws = self.pr_ws[mask]
        return float(np.mean(np.abs(pr_ws - ob_ws)))

    def get_ws_rmse(
        self, wd_grade: typing.Optional[int] = None
   ) -> float:
        """
        Compute root-mean-square wind-speed error.

        Args:
            wd_grade: Wind-direction grade (0-7) to filter observations, or
                None to use all samples.

        Returns:
            Root-mean-square wind-speed error in m/s.

        Raises:
            AssertionError: If wd_grade is not None or an integer 0-7.
        """
        assert wd_grade in (None, 0, 1, 2, 3, 4, 5, 6, 7)
        mask = (~np.isnan(self.ob_ws)) & (~np.isnan(self.pr_ws))
        if wd_grade is not None:
            mask &= self.ob_wd_grade == wd_grade
        ob_ws = self.ob_ws[mask]
        pr_ws = self.pr_ws[mask]
        return float(np.mean((pr_ws - ob_ws) ** 2) ** 0.5)

    def get_ws_mre(
        self, wd_grade: typing.Optional[int] = None
   ) -> float:
        """
        Compute mean relative wind-speed error.

        Args:
            wd_grade: Wind-direction grade (0-7) to filter observations, or
                None to use all samples.

        Returns:
            Mean relative wind-speed error.

        Raises:
            AssertionError: If wd_grade is not None or an integer 0-7.
        """
        assert wd_grade in (None, 0, 1, 2, 3, 4, 5, 6, 7)
        mask = self.pr_ws + self.ob_ws > 0
        if wd_grade is not None:
            mask &= self.ob_wd_grade == wd_grade
        ob_ws = self.ob_ws[mask]
        pr_ws = self.pr_ws[mask]
        return float(np.mean(np.abs((pr_ws - ob_ws) / (pr_ws + ob_ws))))

    def get_ws_r(
        self, wd_grade: typing.Optional[int] = None
   ) -> float:
        """
        Compute Pearson correlation of wind speed.

        Args:
            wd_grade: Wind-direction grade (0-7) to filter observations, or
                None to use all samples.

        Returns:
            Pearson correlation coefficient.

        Raises:
            AssertionError: If wd_grade is not None or an integer 0-7.
        """
        assert wd_grade in (None, 0, 1, 2, 3, 4, 5, 6, 7)
        mask = (~np.isnan(self.ob_ws)) & (~np.isnan(self.pr_ws))
        if wd_grade is not None:
            mask &= self.ob_wd_grade == wd_grade
        ob_ws = self.ob_ws[mask]
        pr_ws = self.pr_ws[mask]
        return float(np.corrcoef(ob_ws, pr_ws)[0, 1])

    def get_ws_ts(self) -> np.ndarray:
        """
        Compute threat score (TS) for each wind-speed grade.

        Returns:
            Threat score for each wind-speed grade.
        """
        ts = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum((self.pr_ws_grade == i) & (self.ob_ws_grade == i))
            nb = np.sum((self.pr_ws_grade == i) & (self.ob_ws_grade != i))
            nc = np.sum((self.pr_ws_grade != i) & (self.ob_ws_grade == i))
            ts[i] = na / (na + nb + nc) if na + nb + nc != 0 else np.nan
        return ts

    def get_ws_ts2(self) -> np.ndarray:
        """
        Compute cumulative threat score (TS2) for wind-speed thresholds.

        Returns:
            Cumulative threat score for each wind-speed threshold.
        """
        ts = np.zeros(self.n_grades - 1, dtype=np.float32)
        for i in range(self.n_grades - 1):
            na = np.sum((self.pr_ws_grade > i) & (self.ob_ws_grade > i))
            nb = np.sum((self.pr_ws_grade > i) & (self.ob_ws_grade <= i))
            nc = np.sum((self.pr_ws_grade <= i) & (self.ob_ws_grade > i))
            ts[i] = na / (na + nb + nc) if na + nb + nc != 0 else np.nan
        return ts

    def get_ws_ets(self) -> np.ndarray:
        """
        Compute equitable threat score (ETS) for each wind-speed grade.

        Returns:
            Equitable threat score for each wind-speed grade.
        """
        ets = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum((self.pr_ws_grade == i) & (self.ob_ws_grade == i))
            nb = np.sum((self.pr_ws_grade == i) & (self.ob_ws_grade != i))
            nc = np.sum((self.pr_ws_grade != i) & (self.ob_ws_grade == i))
            nd = np.sum((self.pr_ws_grade != i) & (self.ob_ws_grade != i))
            expected = (na + nb) / (na + nb + nc + nd) * (na + nc)
            ets[i] = (
                (na - expected) / (na + nb + nc - expected)
                if na + nb + nc != 0
                else np.nan
            )
        return ets

    def get_ws_ets2(self) -> np.ndarray:
        """
        Compute cumulative ETS (ETS2) for wind-speed thresholds.

        Returns:
            Cumulative equitable threat score for each threshold.
        """
        ets = np.zeros(self.n_grades - 1, dtype=np.float32)
        for i in range(self.n_grades - 1):
            na = np.sum((self.pr_ws_grade > i) & (self.ob_ws_grade > i))
            nb = np.sum((self.pr_ws_grade > i) & (self.ob_ws_grade <= i))
            nc = np.sum((self.pr_ws_grade <= i) & (self.ob_ws_grade > i))
            nd = np.sum((self.pr_ws_grade <= i) & (self.ob_ws_grade <= i))
            expected = (na + nb) / (na + nb + nc + nd) * (na + nc)
            ets[i] = (
                (na - expected) / (na + nb + nc - expected)
                if na + nb + nc != 0
                else np.nan
            )
        return ets

    def get_ws_bias(self) -> np.ndarray:
        """
        Compute frequency bias for each wind-speed grade.

        Returns:
            Frequency bias for each wind-speed grade.
        """
        bias = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum((self.pr_ws_grade == i) & (self.ob_ws_grade == i))
            nb = np.sum((self.pr_ws_grade == i) & (self.ob_ws_grade != i))
            nc = np.sum((self.pr_ws_grade != i) & (self.ob_ws_grade == i))
            bias[i] = (na + nb) / (na + nc) if na + nc != 0 else np.nan
        return bias

    def get_ws_far(self) -> np.ndarray:
        """
        Compute false alarm ratio (FAR) for each wind-speed grade.

        Returns:
            False alarm ratio for each wind-speed grade.
        """
        far = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum((self.pr_ws_grade == i) & (self.ob_ws_grade == i))
            nb = np.sum((self.pr_ws_grade == i) & (self.ob_ws_grade != i))
            far[i] = nb / (na + nb) if na + nb != 0 else np.nan
        return far

    def get_ws_mar(self) -> np.ndarray:
        """
        Compute miss ratio (MAR) for each wind-speed grade.

        Returns:
            Miss ratio for each wind-speed grade.
        """
        mar = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum(
                (self.pr_ws_grade == i + 1) & (self.ob_ws_grade == i + 1)
            )
            nc = np.sum(
                (self.pr_ws_grade != i + 1) & (self.ob_ws_grade == i + 1)
            )
            mar[i] = nc / (na + nc) if na + nc != 0 else np.nan
        return mar

    def get_ws_cm(self) -> np.ndarray:
        """
        Return the wind-speed confusion matrix.

        Returns:
            Wind-speed confusion matrix with shape (n_grades + 1, n_grades + 1).
        """
        return self.ws_cm

    def get_ws_oa(self) -> float:
        """
        Compute overall accuracy (OA) for wind-speed grades.

        Returns:
            Overall accuracy for wind-speed grades.
        """
        return np.sum(np.diag(self.ws_cm)) / self.ws_n

    def get_ws_kappa(self) -> float:
        """
        Compute Cohen's kappa for wind-speed grades.

        Returns:
            Cohen's kappa for wind-speed grades.
        """
        row_total = np.sum(self.ws_cm, axis=1).astype(np.float32)
        col_total = np.sum(self.ws_cm, axis=0).astype(np.float32)
        pe = np.sum(row_total * col_total) / self.ws_n / self.ws_n
        return (self.get_ws_oa() - pe) / (1 - pe)



    def copy(self) -> 'Acc':
        """Return a deep copy of this verification object."""
        return copy.deepcopy(self)


def _load_data(
    cfg: typing.Dict[str, typing.Any]
) -> typing.Tuple[
    typing.List[int], pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Load observations and predictions for the four main models."""
    # 1. Load configuration and data.
    cfg = utils.load_config()
    years = cfg['constants']['years']
    base_path = pathlib.Path(cfg['paths']['project_wind'])
    path_97 = base_path / '97'
    path_97uv = base_path / '97-uv'
    fig_dir = path_97uv / '图'
    os.makedirs(fig_dir, exist_ok=True)
    station_csv = pathlib.Path(cfg['paths']['station_csv'])

    # 1.1 Load 97-station wind-speed data and inject into UV arrays.
    ob = list()
    ec = list()
    pdfm = list()
    mlr = list()
    lgb = list()
    for year in years:
        ob.append(np.load(path_97 / f'ob_1_{year}.npy'))
        ec.append(np.load(path_97 / f'nwp_1_{year}.npy'))
        pdfm.append(np.load(path_97 / f'pdfm_4_{year}.npy'))
        mlr.append(np.load(path_97 / f'mlr_4_{year}.npy'))
        lgb.append(np.load(path_97 / f'lgb_3_{year}.npy'))
    ob = np.vstack(ob)
    ec = np.vstack(ec)
    pdfm = np.vstack(pdfm)
    mlr = np.vstack(mlr)
    lgb = np.vstack(lgb)

    ob2 = list()
    ec2 = list()
    pdfm2 = list()
    mlr2 = list()
    lgb2 = list()
    for year in years:
        ob2.append(np.load(path_97uv / f'ob_{year}.npy'))
        ec2.append(np.load(path_97uv / f'nwp_{year}.npy'))
        pdfm2.append(np.load(path_97uv / f'pdfm_3_{year}.npy'))
        mlr2.append(np.load(path_97uv / f'mlr_3_{year}.npy'))
        lgb2.append(np.load(path_97uv / f'lgb_3_{year}.npy'))
    ob2 = np.vstack(ob2)
    ec2 = np.vstack(ec2)
    pdfm2 = np.vstack(pdfm2)
    mlr2 = np.vstack(mlr2)
    lgb2 = np.vstack(lgb2)
    # ob2[:, :, :, 1] = ob
    # ec2[:, :, :, 1] = ec
    # pdfm2[:, :, :, 1] = pdfm
    # mlr2[:, :, :, 1] = mlr
    # lgb2[:, :, :, 1] = lgb

    return (
        years, path_97, path_97uv, fig_dir, station_csv,
        ob2, ec2, pdfm2, mlr2, lgb2
    )


def _compute_overall_scores(
    ob2: np.ndarray, ec2: np.ndarray, pdfm2: np.ndarray,
    mlr2: np.ndarray, lgb2: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute overall and wind-direction-dependent verification scores."""
    # 2. Compute overall scores and direction-dependent scores.
    ts_grade = np.zeros((2, 4, 8), dtype=np.float32) + np.nan
    wd_kappa_grade = np.zeros((4, 8), dtype=np.float32) + np.nan
    ws_rmse_wd = np.zeros((4, 9), dtype=np.float32) + np.nan
    wd_rmse_wd = np.zeros((4, 9), dtype=np.float32) + np.nan

    models = [
        ('ECMWF-IFS', ec2),
        ('PDFM', pdfm2),
        ('MOS', mlr2),
        ('LightGBM', lgb2),
    ]
    for idx, (name, pred) in enumerate(models):
        print(f'----- {name} -----')
        acc = Acc(ob2, pred)
        print(f'  风速 ME    : {acc.get_ws_me():.4f} m/s')
        print(f'  风速 CC     : {acc.get_ws_r():.4f}')
        print(f'  风速 MAE   : {acc.get_ws_mae():.4f} m/s')
        print(f'  风速 RMSE  : {acc.get_ws_rmse():.4f} m/s')
        print(f'  风速 MRE   : {acc.get_ws_mre():.4f}')
        print(f'  风速 OA    : {acc.get_ws_oa():.4f}')
        print(f'  风速 kappa : {acc.get_ws_kappa():.4f}')
        ts_grade[0, idx, :] = acc.get_ws_ts()[1:]
        ts_grade[1, idx, :] = acc.get_ws_ts2()
        wd_kappa_grade[idx, :] = acc.get_wd_kappa(True)
        print(f'  风向 ME   : {acc.get_wd_me():.2f} °')
        print(f'  风向 MAE   : {acc.get_wd_mae():.2f} °')
        print(f'  风向 RMSE  : {acc.get_wd_rmse():.2f} °')
        print(f'  风向 OA    : {acc.get_wd_oa():.4f}')
        print(f'  风向 kappa : {acc.get_wd_kappa():.4f}')
        for i, j in enumerate([2, 1, 0, 7, 6, 5, 4, 3, 2]):
            wd_rmse_wd[idx, i] = acc.get_wd_rmse(wd_grade=j)
            ws_rmse_wd[idx, i] = acc.get_ws_rmse(wd_grade=j)
    return ts_grade, wd_kappa_grade, ws_rmse_wd, wd_rmse_wd


def _build_time_indices(
    years: typing.List[int]
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build year, month, and forecast-hour indices for grouping."""
    # 4. Build time indices for year/month/forecast-hour grouping.
    n_cases = round(
        (arrow.get(str(years[-1] + 1)) - arrow.get(str(years[0]))).total_seconds()
        / 3600 / 12
    )
    year_ind = np.zeros(n_cases, np.int_) - 1
    month_ind = np.zeros(n_cases, np.int_) - 1
    fhour_ind = np.zeros((n_cases, 24), np.int_) - 1
    for i in range(n_cases):
        time_arrow = arrow.get(str(years[0])).shift(hours=12 * i)
        year_ind[i] = time_arrow.datetime.year
        month_ind[i] = time_arrow.shift(hours=44).datetime.month
        for j in range(24):
            fhour_ind[i, j] = time_arrow.shift(hours=13 + j).datetime.hour
    return year_ind, month_ind, fhour_ind

def _compute_yearly_rmse(
    ob2: np.ndarray, ec2: np.ndarray, pdfm2: np.ndarray,
    mlr2: np.ndarray, lgb2: np.ndarray, year_ind: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Compute yearly wind-speed and wind-direction RMSE."""
    # 4.1 RMSE by year.
    ws_rmse_year = np.zeros((4, 6), dtype=np.float32) + np.nan
    wd_rmse_year = np.zeros((4, 6), dtype=np.float32) + np.nan
    for i in range(6):
        mask = year_ind == i + 2017
        acc = Acc(ob2[mask, :, :], ec2[mask, :, :])
        ws_rmse_year[0, i] = acc.get_ws_rmse()
        wd_rmse_year[0, i] = acc.get_wd_rmse()
        acc = Acc(ob2[mask, :, :], pdfm2[mask, :, :])
        ws_rmse_year[1, i] = acc.get_ws_rmse()
        wd_rmse_year[1, i] = acc.get_wd_rmse()
        acc = Acc(ob2[mask, :, :], mlr2[mask, :, :])
        ws_rmse_year[2, i] = acc.get_ws_rmse()
        wd_rmse_year[2, i] = acc.get_wd_rmse()
        acc = Acc(ob2[mask, :, :], lgb2[mask, :, :])
        ws_rmse_year[3, i] = acc.get_ws_rmse()
        wd_rmse_year[3, i] = acc.get_wd_rmse()
    return ws_rmse_year, wd_rmse_year


def _compute_monthly_rmse(
    ob2: np.ndarray, ec2: np.ndarray, pdfm2: np.ndarray,
    mlr2: np.ndarray, lgb2: np.ndarray, month_ind: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Compute monthly wind-speed and wind-direction RMSE."""
    # 4.2 RMSE by month.
    ws_rmse_month = np.zeros((4, 12), dtype=np.float32) + np.nan
    wd_rmse_month = np.zeros((4, 12), dtype=np.float32) + np.nan
    for i in range(12):
        mask = month_ind == i + 1
        acc = Acc(ob2[mask, :, :], ec2[mask, :, :])
        ws_rmse_month[0, i] = acc.get_ws_rmse()
        wd_rmse_month[0, i] = acc.get_wd_rmse()
        acc = Acc(ob2[mask, :, :], pdfm2[mask, :, :])
        ws_rmse_month[1, i] = acc.get_ws_rmse()
        wd_rmse_month[1, i] = acc.get_wd_rmse()
        acc = Acc(ob2[mask, :, :], mlr2[mask, :, :])
        ws_rmse_month[2, i] = acc.get_ws_rmse()
        wd_rmse_month[2, i] = acc.get_wd_rmse()
        acc = Acc(ob2[mask, :, :], lgb2[mask, :, :])
        ws_rmse_month[3, i] = acc.get_ws_rmse()
        wd_rmse_month[3, i] = acc.get_wd_rmse()
    i = int(np.argmin(ws_rmse_month[3, :]))
    print(f'LightGBM 逐月风速RMSE最小: {i + 1}月, {ws_rmse_month[3, i]:.4f} m/s')
    i = int(np.argmax(ws_rmse_month[3, :]))
    print(f'LightGBM 逐月风速RMSE最大: {i + 1}月, {ws_rmse_month[3, i]:.4f} m/s')
    i = int(np.argmin(wd_rmse_month[3, :]))
    print(f'LightGBM 逐月风向RMSE最小: {i + 1}月, {wd_rmse_month[3, i]:.2f} °')
    i = int(np.argmax(wd_rmse_month[3, :]))
    print(f'LightGBM 逐月风向RMSE最大: {i + 1}月, {wd_rmse_month[3, i]:.2f} °')
    return ws_rmse_month, wd_rmse_month


def _compute_leadtime_rmse(
    ob2: np.ndarray, ec2: np.ndarray, pdfm2: np.ndarray,
    mlr2: np.ndarray, lgb2: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Compute forecast-lead-time wind-speed and wind-direction RMSE."""
    # 4.3 RMSE by forecast lead time.
    ws_rmse_vt = np.zeros((4, 24), dtype=np.float32) + np.nan
    wd_rmse_vt = np.zeros((4, 24), dtype=np.float32) + np.nan
    for i in range(24):
        acc = Acc(ob2[:, i, :], ec2[:, i, :])
        ws_rmse_vt[0, i] = acc.get_ws_rmse()
        wd_rmse_vt[0, i] = acc.get_wd_rmse()
        acc = Acc(ob2[:, i, :], pdfm2[:, i, :])
        ws_rmse_vt[1, i] = acc.get_ws_rmse()
        wd_rmse_vt[1, i] = acc.get_wd_rmse()
        acc = Acc(ob2[:, i, :], mlr2[:, i, :])
        ws_rmse_vt[2, i] = acc.get_ws_rmse()
        wd_rmse_vt[2, i] = acc.get_wd_rmse()
        acc = Acc(ob2[:, i, :], lgb2[:, i, :])
        ws_rmse_vt[3, i] = acc.get_ws_rmse()
        wd_rmse_vt[3, i] = acc.get_wd_rmse()
    return ws_rmse_vt, wd_rmse_vt


def _compute_forecast_hour_rmse(
    ob2: np.ndarray, ec2: np.ndarray, pdfm2: np.ndarray,
    mlr2: np.ndarray, lgb2: np.ndarray, fhour_ind: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Compute forecast-hour-of-day wind-speed and wind-direction RMSE."""
    # 4.4 RMSE by forecast hour of day.
    ws_rmse_fhour = np.zeros((4, 24), dtype=np.float32) + np.nan
    wd_rmse_fhour = np.zeros((4, 24), dtype=np.float32) + np.nan
    for i in range(24):
        mask = fhour_ind == i
        acc = Acc(ob2[mask], ec2[mask])
        ws_rmse_fhour[0, i] = acc.get_ws_rmse()
        wd_rmse_fhour[0, i] = acc.get_wd_rmse()
        acc = Acc(ob2[mask], pdfm2[mask])
        ws_rmse_fhour[1, i] = acc.get_ws_rmse()
        wd_rmse_fhour[1, i] = acc.get_wd_rmse()
        acc = Acc(ob2[mask], mlr2[mask])
        ws_rmse_fhour[2, i] = acc.get_ws_rmse()
        wd_rmse_fhour[2, i] = acc.get_wd_rmse()
        acc = Acc(ob2[mask], lgb2[mask])
        ws_rmse_fhour[3, i] = acc.get_ws_rmse()
        wd_rmse_fhour[3, i] = acc.get_wd_rmse()
    i = int(np.argmin(ws_rmse_fhour[3, :]))
    print(f'LightGBM 逐时次风速RMSE最小: {i}时, {ws_rmse_fhour[3, i]:.4f} m/s')
    i = int(np.argmax(ws_rmse_fhour[3, :]))
    print(f'LightGBM 逐时次风速RMSE最大: {i}时, {ws_rmse_fhour[3, i]:.4f} m/s')
    i = int(np.argmin(wd_rmse_fhour[3, :]))
    print(f'LightGBM 逐时次风向RMSE最小: {i}时, {wd_rmse_fhour[3, i]:.2f} °')
    i = int(np.argmax(wd_rmse_fhour[3, :]))
    print(f'LightGBM 逐时次风向RMSE最大: {i}时, {wd_rmse_fhour[3, i]:.2f} °')
    return ws_rmse_fhour, wd_rmse_fhour


def main() -> None:
    '''Run the full evaluation and plotting workflow.'''
    cfg = utils.load_config()
    (
        years, path_97, path_97uv, fig_dir, station_csv,
        ob2, ec2, pdfm2, mlr2, lgb2
    ) = _load_data(cfg)

    mi = np.load(path_97uv / 'mi.npy')
    access97_uv_plots._plot_mi_heatmap(mi, fig_dir)

    ts_grade, wd_kappa_grade, ws_rmse_wd, wd_rmse_wd = _compute_overall_scores(
        ob2, ec2, pdfm2, mlr2, lgb2
    )
    access97_uv_plots._plot_grade_bars(ts_grade, wd_kappa_grade, fig_dir)

    year_ind, month_ind, fhour_ind = _build_time_indices(years)

    ws_rmse_year, wd_rmse_year = _compute_yearly_rmse(
        ob2, ec2, pdfm2, mlr2, lgb2, year_ind
    )
    access97_uv_plots._plot_yearly_rmse(ws_rmse_year, wd_rmse_year, fig_dir)

    ws_rmse_month, wd_rmse_month = _compute_monthly_rmse(
        ob2, ec2, pdfm2, mlr2, lgb2, month_ind
    )
    access97_uv_plots._plot_monthly_rmse(ws_rmse_month, wd_rmse_month, fig_dir)

    ws_rmse_vt, wd_rmse_vt = _compute_leadtime_rmse(
        ob2, ec2, pdfm2, mlr2, lgb2
    )
    access97_uv_plots._plot_leadtime_rmse(ws_rmse_vt, wd_rmse_vt, fig_dir)

    ws_rmse_fhour, wd_rmse_fhour = _compute_forecast_hour_rmse(
        ob2, ec2, pdfm2, mlr2, lgb2, fhour_ind
    )
    access97_uv_plots._plot_forecast_hour_rmse(
        ws_rmse_fhour, wd_rmse_fhour, fig_dir
    )

    access97_uv_plots._analyze_stations(
        ob2, ec2, pdfm2, mlr2, lgb2, station_csv, fig_dir
    )

    access97_uv_plots._plot_radar(ws_rmse_wd, wd_rmse_wd, fig_dir)

    access97_uv_plots._compare_lgb_variants(ob2, path_97, path_97uv, years)

def _interp(data: np.ndarray, is_0_84: bool = False) -> np.ndarray:
    """
    Interpolate forecast lead times to hourly values.

    Args:
        data: Input array with lead times along the second axis.
        is_0_84: Whether to apply the 0-84h interpolation scheme.

    Returns:
        Array with interpolated lead times.
    """
    data = data.copy()
    if is_0_84:
        for i in range(24):
            data[:, 3 * i + 1, :, :] = (
                data[:, 3 * i, :, :] * 2 / 3
                + data[:, 3 * i + 3, :, :] / 3
            )
            data[:, 3 * i + 2, :, :] = (
                data[:, 3 * i, :, :] / 3
                + data[:, 3 * i + 3, :, :] * 2 / 3
            )
        data[:, 73, :, :] = (
            data[:, 72, :, :] * 5 / 6
            + data[:, 78, :, :] / 6
        )
        data[:, 74, :, :] = (
            data[:, 72, :, :] * 2 / 3
            + data[:, 78, :, :] / 3
        )
        data[:, 75, :, :] = (
            data[:, 72, :, :] / 2
            + data[:, 78, :, :] / 2
        )
        data[:, 76, :, :] = (
            data[:, 72, :, :] / 3
            + data[:, 78, :, :] * 2 / 3
        )
        data[:, 77, :, :] = (
            data[:, 72, :, :] / 6
            + data[:, 78, :, :] * 5 / 6
        )
        data[:, 79, :, :] = (
            data[:, 78, :, :] * 5 / 6
            + data[:, 84, :, :] / 6
        )
        data[:, 80, :, :] = (
            data[:, 78, :, :] * 2 / 3
            + data[:, 84, :, :] / 3
        )
        data[:, 81, :, :] = (
            data[:, 78, :, :] / 2
            + data[:, 84, :, :] / 2
        )
        data[:, 82, :, :] = (
            data[:, 78, :, :] / 3
            + data[:, 84, :, :] * 2 / 3
        )
        data[:, 83, :, :] = (
            data[:, 78, :, :] / 6
            + data[:, 84, :, :] * 5 / 6
        )
    else:
        for i in range(8):
            data[:, 3 * i + 1, :, :] = (
                data[:, 3 * i, :, :] * 2 / 3
                + data[:, 3 * i + 3, :, :] / 3
            )
            data[:, 3 * i + 2, :, :] = (
                data[:, 3 * i, :, :] / 3
                + data[:, 3 * i + 3, :, :] * 2 / 3
            )
    return data


if __name__ == '__main__':
    print('Program access97_uv.py started')
    total_start = arrow.now()

    main()

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(
        'Program access97_uv.py finished, total time '
        f'{utils.format_time(total_elapsed)}'
    )
