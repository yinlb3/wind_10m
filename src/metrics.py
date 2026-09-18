#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wind verification metrics module

Provides the Acc class for wind-direction and wind-speed verification
scores, shared by the 97-station evaluation scripts and the plotting
helpers under evaluation/.

Founded in 2026-09-18
Modified in 2026-09-18
@author: yinlb, deepseek-flash
"""

import copy
import typing

import numpy as np


# Wind-speed thresholds for Beaufort-like grades (m/s)
THRES = (0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2)


class Acc:
    """
    Wind-direction and wind-speed verification metrics.

    The wind-direction channel is optional: construct with ``has_wd=False``
    to obtain a wind-speed-only verification object, in which case the
    wind-direction attributes stay None and the wind-direction scores must
    not be used.

    Attributes:
        _has_wd (bool): Whether wind-direction data is available.
        _thres (tuple): Wind-speed thresholds for Beaufort-like grades.
        _n_grades (int): Number of wind-speed grades, equals len(thres) + 1.
        _ob_wd (np.ndarray): Observed wind direction.
        _pr_wd (np.ndarray): Predicted wind direction.
        _ob_wd_grade (np.ndarray): Observed wind-direction grade (0-7).
        _pr_wd_grade (np.ndarray): Predicted wind-direction grade (0-7).
        _wd_cm (np.ndarray): Confusion matrix for wind-direction grades.
        _wd_n (int): Total number of wind-direction samples.
        _ob_ws (np.ndarray): Observed wind speed.
        _pr_ws (np.ndarray): Predicted wind speed.
        _ob_ws_grade (np.ndarray): Observed wind-speed grade.
        _pr_ws_grade (np.ndarray): Predicted wind-speed grade.
        _ws_cm (np.ndarray): Confusion matrix for wind-speed grades.
        _ws_n (int): Total number of wind-speed samples.
    """

    def __init__(
        self, ob: np.ndarray, pr: np.ndarray, has_wd: bool = True
    ) -> None:
        """
        Initialize verification object from observation and prediction.

        Args:
            ob (np.ndarray): Observation array. When ``has_wd`` is True the
                last dimension holds (wind direction, wind speed); otherwise
                it holds wind speed only.
            pr (np.ndarray): Prediction array with the same shape as ob.
            has_wd (bool): Whether the input carries a wind-direction
                channel. Defaults to True. The wind-direction scores are
                unavailable when False.
        """
        self._has_wd = has_wd
        self._thres = THRES
        self._n_grades = len(self._thres) + 1

        if has_wd:
            self._ob_wd = ob[..., 0]
            self._pr_wd = pr[..., 0]

            self._ob_wd_grade = np.zeros_like(self._ob_wd, dtype=np.int_) - 1
            for i in range(8):
                if i == 0:
                    mask = (self._ob_wd < 22.5) | (self._ob_wd >= 337.5)
                    self._ob_wd_grade[mask] = i
                else:
                    mask = (
                        (self._ob_wd >= 45 * i - 22.5)
                        & (self._ob_wd < 45 * i + 22.5)
                    )
                    self._ob_wd_grade[mask] = i

            self._pr_wd_grade = np.zeros_like(self._pr_wd, dtype=np.int_) - 1
            for i in range(8):
                if i == 0:
                    mask = (
                        ((self._pr_wd < 22.5) & (self._pr_wd >= 0))
                        | ((self._pr_wd >= 337.5) & (self._pr_wd <= 360))
                    )
                    self._pr_wd_grade[mask] = i
                else:
                    mask = (
                        (self._pr_wd >= 45 * i - 22.5)
                        & (self._pr_wd < 45 * i + 22.5)
                    )
                    self._pr_wd_grade[mask] = i

            self._wd_cm = np.zeros((8, 8), dtype=np.int_)
            for i in range(8):
                for j in range(8):
                    self._wd_cm[i, j] = np.sum(
                        (self._ob_wd_grade == i) & (self._pr_wd_grade == j)
                    )
            self._wd_n = np.sum(self._wd_cm)
        else:
            self._ob_wd = None
            self._pr_wd = None
            self._ob_wd_grade = None
            self._pr_wd_grade = None
            self._wd_cm = None
            self._wd_n = 0

        self._ob_ws = ob[..., 1] if has_wd else ob
        self._pr_ws = pr[..., 1] if has_wd else pr

        self._ob_ws_grade = np.zeros_like(self._ob_ws, dtype=np.int_) - 1
        self._ob_ws_grade[~np.isnan(self._ob_ws)] = 0
        for i in range(self._n_grades - 1):
            self._ob_ws_grade[self._ob_ws >= self._thres[i]] = i + 1

        self._pr_ws_grade = np.zeros_like(self._pr_ws, dtype=np.int_) - 1
        self._pr_ws_grade[~np.isnan(self._pr_ws)] = 0
        for i in range(self._n_grades - 1):
            self._pr_ws_grade[self._pr_ws >= self._thres[i]] = i + 1

        self._ws_cm = np.zeros(
            (self._n_grades + 1, self._n_grades + 1), dtype=np.int_
        )
        for i in range(self._n_grades + 1):
            for j in range(self._n_grades + 1):
                self._ws_cm[i, j] = np.sum(
                    (self._ob_ws_grade == i) & (self._pr_ws_grade == j)
                )
        self._ws_n = np.sum(self._ws_cm)

    def _get_wd_delta(
        self, wd_grade: typing.Optional[int] = None
    ) -> np.ndarray:
        """
        Compute wind-direction error array after circular adjustment.

        Args:
            wd_grade (typing.Optional[int]): Wind-direction grade (0-7) to
                filter observations, or None to use all samples.

        Returns:
            Wind-direction error array in degrees, range [-180, 180].

        Raises:
            AssertionError: If wd_grade is not None or an integer 0-7.
        """
        assert wd_grade in (None, 0, 1, 2, 3, 4, 5, 6, 7)
        mask = (~np.isnan(self._ob_wd)) & (~np.isnan(self._pr_wd))
        if wd_grade is not None:
            mask &= self._ob_wd_grade == wd_grade
        ob_wd = self._ob_wd[mask]
        pr_wd = self._pr_wd[mask]
        delta = pr_wd - ob_wd
        keep = (ob_wd != 999017) & (pr_wd != 999017)
        delta = delta[keep]
        delta[delta > 180] -= 360
        delta[delta < -180] += 360
        return delta

    def get_wd_me(
        self, wd_grade: typing.Optional[int] = None
    ) -> float:
        """
        Compute mean wind-direction error.

        Args:
            wd_grade (typing.Optional[int]): Wind-direction grade (0-7) to
                filter observations, or None to use all samples.

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
            wd_grade (typing.Optional[int]): Wind-direction grade (0-7) to
                filter observations, or None to use all samples.

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
            wd_grade (typing.Optional[int]): Wind-direction grade (0-7) to
                filter observations, or None to use all samples.

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
            wd_grade (typing.Optional[int]): Wind-direction grade (0-7) to
                filter observations, or None to use all samples.

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
        return self._wd_cm

    def get_wd_oa(
        self, is_ws: bool = False
    ) -> typing.Union[float, np.ndarray]:
        """
        Compute overall accuracy (OA) for wind-direction grades.

        Args:
            is_ws (bool): Whether to compute OA per wind-speed grade.

        Returns:
            Scalar OA when is_ws is False, otherwise an array of OA values.
        """
        if is_ws:
            oa = np.zeros(self._n_grades - 1, dtype=np.float32) + np.nan
            for i in range(1, self._n_grades):
                wd_cm = np.zeros((8, 8), dtype=np.int_)
                for j in range(8):
                    for k in range(8):
                        mask = (
                            (self._ob_ws_grade == i)
                            & (self._ob_wd_grade == j)
                            & (self._pr_wd_grade == k)
                        )
                        wd_cm[j, k] = np.sum(mask)
                oa[i - 1] = np.sum(np.diag(wd_cm)) / np.sum(wd_cm)
        else:
            oa = np.sum(np.diag(self._wd_cm)) / self._wd_n
        return oa

    def get_wd_kappa(
        self, is_ws: bool = False
    ) -> typing.Union[float, np.ndarray]:
        """
        Compute Cohen's kappa for wind-direction grades.

        Args:
            is_ws (bool): Whether to compute kappa per wind-speed grade.

        Returns:
            Scalar kappa when is_ws is False, otherwise an array of kappa.
        """
        if is_ws:
            kappa0 = np.zeros(self._n_grades - 1, dtype=np.float32) + np.nan
            for i in range(1, self._n_grades):
                wd_cm = np.zeros((8, 8), dtype=np.int_)
                for j in range(8):
                    for k in range(8):
                        mask = (
                            (self._ob_ws_grade == i)
                            & (self._ob_wd_grade == j)
                            & (self._pr_wd_grade == k)
                        )
                        wd_cm[j, k] = np.sum(mask)
                row_total = np.sum(wd_cm, axis=1).astype(np.float32)
                col_total = np.sum(wd_cm, axis=0).astype(np.float32)
                total = np.sum(wd_cm)
                pe = np.sum(row_total * col_total) / total / total
                oa = np.sum(np.diag(wd_cm)) / total
                kappa0[i - 1] = (oa - pe) / (1 - pe)
        else:
            row_total = np.sum(self._wd_cm, axis=1).astype(np.float32)
            col_total = np.sum(self._wd_cm, axis=0).astype(np.float32)
            pe = np.sum(row_total * col_total) / self._wd_n / self._wd_n
            kappa0 = (self.get_wd_oa() - pe) / (1 - pe)
        return kappa0

    def get_ws_me(self) -> float:
        """
        Compute mean wind-speed error.

        Returns:
            Mean wind-speed error (prediction minus observation) in m/s.
        """
        mask = (~np.isnan(self._ob_ws)) & (~np.isnan(self._pr_ws))
        ob_ws = self._ob_ws[mask]
        pr_ws = self._pr_ws[mask]
        return float(np.mean(pr_ws - ob_ws))

    def get_ws_mae(
        self, wd_grade: typing.Optional[int] = None
    ) -> float:
        """
        Compute mean absolute wind-speed error.

        Args:
            wd_grade (typing.Optional[int]): Wind-direction grade (0-7) to
                filter observations, or None to use all samples.

        Returns:
            Mean absolute wind-speed error in m/s.

        Raises:
            AssertionError: If wd_grade is not None or an integer 0-7.
        """
        assert wd_grade in (None, 0, 1, 2, 3, 4, 5, 6, 7)
        mask = (~np.isnan(self._ob_ws)) & (~np.isnan(self._pr_ws))
        if wd_grade is not None:
            mask &= self._ob_wd_grade == wd_grade
        ob_ws = self._ob_ws[mask]
        pr_ws = self._pr_ws[mask]
        return float(np.mean(np.abs(pr_ws - ob_ws)))

    def get_ws_rmse(
        self, wd_grade: typing.Optional[int] = None
    ) -> float:
        """
        Compute root-mean-square wind-speed error.

        Args:
            wd_grade (typing.Optional[int]): Wind-direction grade (0-7) to
                filter observations, or None to use all samples.

        Returns:
            Root-mean-square wind-speed error in m/s.

        Raises:
            AssertionError: If wd_grade is not None or an integer 0-7.
        """
        assert wd_grade in (None, 0, 1, 2, 3, 4, 5, 6, 7)
        mask = (~np.isnan(self._ob_ws)) & (~np.isnan(self._pr_ws))
        if wd_grade is not None:
            mask &= self._ob_wd_grade == wd_grade
        ob_ws = self._ob_ws[mask]
        pr_ws = self._pr_ws[mask]
        return float(np.mean((pr_ws - ob_ws) ** 2) ** 0.5)

    def get_ws_mre(
        self, wd_grade: typing.Optional[int] = None
    ) -> float:
        """
        Compute mean relative wind-speed error.

        Args:
            wd_grade (typing.Optional[int]): Wind-direction grade (0-7) to
                filter observations, or None to use all samples.

        Returns:
            Mean relative wind-speed error.

        Raises:
            AssertionError: If wd_grade is not None or an integer 0-7.
        """
        assert wd_grade in (None, 0, 1, 2, 3, 4, 5, 6, 7)
        mask = self._pr_ws + self._ob_ws > 0
        if wd_grade is not None:
            mask &= self._ob_wd_grade == wd_grade
        ob_ws = self._ob_ws[mask]
        pr_ws = self._pr_ws[mask]
        return float(np.mean(np.abs((pr_ws - ob_ws) / (pr_ws + ob_ws))))

    def get_ws_fs(self) -> float:
        """
        Compute frequency score for wind-speed grades.

        Returns:
            Frequency score based on the grade differences.
        """
        mask = (~np.isnan(self._ob_ws)) & (~np.isnan(self._pr_ws))
        ob_ws = self._ob_ws_grade[mask]
        pr_ws = self._pr_ws_grade[mask]
        return float(
            100 - 40 * np.mean(np.abs((pr_ws - ob_ws) / len(self._thres)))
        )

    def get_ws_r(
        self, wd_grade: typing.Optional[int] = None
    ) -> float:
        """
        Compute Pearson correlation of wind speed.

        Args:
            wd_grade (typing.Optional[int]): Wind-direction grade (0-7) to
                filter observations, or None to use all samples.

        Returns:
            Pearson correlation coefficient.

        Raises:
            AssertionError: If wd_grade is not None or an integer 0-7.
        """
        assert wd_grade in (None, 0, 1, 2, 3, 4, 5, 6, 7)
        mask = (~np.isnan(self._ob_ws)) & (~np.isnan(self._pr_ws))
        if wd_grade is not None:
            mask &= self._ob_wd_grade == wd_grade
        ob_ws = self._ob_ws[mask]
        pr_ws = self._pr_ws[mask]
        return float(np.corrcoef(ob_ws, pr_ws)[0, 1])

    def get_ws_ts(self) -> np.ndarray:
        """
        Compute threat score (TS) for each wind-speed grade.

        Returns:
            Threat score for each wind-speed grade.
        """
        ts = np.zeros(self._n_grades, dtype=np.float32)
        for i in range(self._n_grades):
            na = np.sum((self._pr_ws_grade == i) & (self._ob_ws_grade == i))
            nb = np.sum((self._pr_ws_grade == i) & (self._ob_ws_grade != i))
            nc = np.sum((self._pr_ws_grade != i) & (self._ob_ws_grade == i))
            ts[i] = na / (na + nb + nc) if na + nb + nc != 0 else np.nan
        return ts

    def get_ws_ts2(self) -> np.ndarray:
        """
        Compute cumulative threat score (TS2) for wind-speed thresholds.

        Returns:
            Cumulative threat score for each wind-speed threshold.
        """
        ts = np.zeros(self._n_grades - 1, dtype=np.float32)
        for i in range(self._n_grades - 1):
            na = np.sum((self._pr_ws_grade > i) & (self._ob_ws_grade > i))
            nb = np.sum((self._pr_ws_grade > i) & (self._ob_ws_grade <= i))
            nc = np.sum((self._pr_ws_grade <= i) & (self._ob_ws_grade > i))
            ts[i] = na / (na + nb + nc) if na + nb + nc != 0 else np.nan
        return ts

    def get_ws_ets(self) -> np.ndarray:
        """
        Compute equitable threat score (ETS) for each wind-speed grade.

        Returns:
            Equitable threat score for each wind-speed grade.
        """
        ets = np.zeros(self._n_grades, dtype=np.float32)
        for i in range(self._n_grades):
            na = np.sum((self._pr_ws_grade == i) & (self._ob_ws_grade == i))
            nb = np.sum((self._pr_ws_grade == i) & (self._ob_ws_grade != i))
            nc = np.sum((self._pr_ws_grade != i) & (self._ob_ws_grade == i))
            nd = np.sum((self._pr_ws_grade != i) & (self._ob_ws_grade != i))
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
        ets = np.zeros(self._n_grades - 1, dtype=np.float32)
        for i in range(self._n_grades - 1):
            na = np.sum((self._pr_ws_grade > i) & (self._ob_ws_grade > i))
            nb = np.sum((self._pr_ws_grade > i) & (self._ob_ws_grade <= i))
            nc = np.sum((self._pr_ws_grade <= i) & (self._ob_ws_grade > i))
            nd = np.sum((self._pr_ws_grade <= i) & (self._ob_ws_grade <= i))
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
        bias = np.zeros(self._n_grades, dtype=np.float32)
        for i in range(self._n_grades):
            na = np.sum((self._pr_ws_grade == i) & (self._ob_ws_grade == i))
            nb = np.sum((self._pr_ws_grade == i) & (self._ob_ws_grade != i))
            nc = np.sum((self._pr_ws_grade != i) & (self._ob_ws_grade == i))
            bias[i] = (na + nb) / (na + nc) if na + nc != 0 else np.nan
        return bias

    def get_ws_far(self) -> np.ndarray:
        """
        Compute false alarm ratio (FAR) for each wind-speed grade.

        Returns:
            False alarm ratio for each wind-speed grade.
        """
        far = np.zeros(self._n_grades, dtype=np.float32)
        for i in range(self._n_grades):
            na = np.sum((self._pr_ws_grade == i) & (self._ob_ws_grade == i))
            nb = np.sum((self._pr_ws_grade == i) & (self._ob_ws_grade != i))
            far[i] = nb / (na + nb) if na + nb != 0 else np.nan
        return far

    def get_ws_mar(self) -> np.ndarray:
        """
        Compute miss ratio (MAR) for each wind-speed grade.

        Returns:
            Miss ratio for each wind-speed grade.
        """
        mar = np.zeros(self._n_grades, dtype=np.float32)
        for i in range(self._n_grades):
            na = np.sum(
                (self._pr_ws_grade == i + 1) & (self._ob_ws_grade == i + 1)
            )
            nc = np.sum(
                (self._pr_ws_grade != i + 1) & (self._ob_ws_grade == i + 1)
            )
            mar[i] = nc / (na + nc) if na + nc != 0 else np.nan
        return mar

    def get_ws_cm(self) -> np.ndarray:
        """
        Return the wind-speed confusion matrix.

        Returns:
            Wind-speed confusion matrix with shape (n_grades + 1, n_grades + 1).
        """
        return self._ws_cm

    def get_ws_oa(self) -> float:
        """
        Compute overall accuracy (OA) for wind-speed grades.

        Returns:
            Overall accuracy for wind-speed grades.
        """
        return np.sum(np.diag(self._ws_cm)) / self._ws_n

    def get_ws_kappa(self) -> float:
        """
        Compute Cohen's kappa for wind-speed grades.

        Returns:
            Cohen's kappa for wind-speed grades.
        """
        row_total = np.sum(self._ws_cm, axis=1).astype(np.float32)
        col_total = np.sum(self._ws_cm, axis=0).astype(np.float32)
        pe = np.sum(row_total * col_total) / self._ws_n / self._ws_n
        return (self.get_ws_oa() - pe) / (1 - pe)

    def copy(self) -> 'Acc':
        """
        Return a deep copy of this verification object.

        Returns:
            Acc: A deep copy of this instance.
        """
        return copy.deepcopy(self)


