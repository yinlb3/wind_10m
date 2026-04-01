#!user/bin.python3

"""
Founded in 2024-10-11
Modified in 2025-09-05
@author: yinlb
"""
import os
import sys
import typing

import arrow
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from meteva import base as meb
from sklearn import feature_selection as fs


THRES = (0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2)
ELEMENT_NAME_LIST = ['T_2m', 'MSL', 'D_2m', 'U_10m', 'V_10m', 'T_1000hPa', 'GH_1000hPa', 'R_1000hPa', 'U_1000hPa',
                     'V_1000hPa', 'T_925hPa', 'GH_925hPa', 'R_925hPa', 'U_925hPa', 'V_925hPa', 'T_850hPa', 'GH_850hPa',
                     'R_850hPa', 'U_850hPa', 'V_850hPa', 'T_700hPa', 'GH_700hPa', 'R_700hPa', 'U_700hPa', 'V_700hPa',
                     'T_500hPa', 'GH_500hPa', 'R_500hPa', 'U_500hPa', 'V_500hPa']


class Acc:
    def __init__(self, ob: np.ndarray, pr: np.ndarray):
        self.thres = THRES
        self.n_grades = len(self.thres) + 1
        self.ob_wd = ob[..., 0]
        self.pr_wd = pr[..., 0]
        self.ob_wd_grade = np.zeros_like(self.ob_wd, dtype=np.int_) - 1
        for i in range(8):
            if i == 0:
                self.ob_wd_grade[(self.ob_wd < 22.5) | (self.ob_wd >= 337.5)] = i
            else:
                self.ob_wd_grade[(self.ob_wd >= 45 * i - 22.5) & (self.ob_wd < 45 * i + 22.5)] = i
        self.pr_wd_grade = np.zeros_like(self.pr_wd, dtype=np.int_) - 1
        for i in range(8):
            if i == 0:
                self.pr_wd_grade[((self.pr_wd < 22.5) & (self.pr_wd >= 0)) | ((self.pr_wd >= 337.5) & (self.pr_wd <= 360))] = i
            else:
                self.pr_wd_grade[(self.pr_wd >= 45 * i - 22.5) & (self.pr_wd < 45 * i + 22.5)] = i
        self.wd_cm = np.zeros((8, 8), dtype=np.int_)
        for i in range(8):
            for j in range(8):
                self.wd_cm[i, j] = np.sum((self.ob_wd_grade == i) & (self.pr_wd_grade == j))
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
        self.ws_cm = np.zeros((self.n_grades + 1, self.n_grades + 1), dtype=np.int_)
        for i in range(self.n_grades + 1):
            for j in range(self.n_grades + 1):
                self.ws_cm[i, j] = np.sum((self.ob_ws_grade == i) & (self.pr_ws_grade == j))
        self.ws_n = np.sum(self.ws_cm)

    def get_wd_me(self, l: typing.Optional[int] = None) -> float:
        assert l in (None, 0, 1, 2, 3, 4, 5, 6, 7)
        index = (~np.isnan(self.ob_wd)) & (~np.isnan(self.pr_wd))
        if l is not None:
            index &= self.ob_wd_grade == l
        ob_wd = self.ob_wd[index]
        pr_wd = self.pr_wd[index]
        delta = pr_wd - ob_wd
        delta = delta[ob_wd != 999017]
        delta[delta > 180] -= 360
        delta[delta < -180] += 360
        return float(np.mean(delta))

    def get_wd_mae(self, l: typing.Optional[int] = None) -> float:
        assert l in (None, 0, 1, 2, 3, 4, 5, 6, 7)
        index = (~np.isnan(self.ob_wd)) & (~np.isnan(self.pr_wd))
        if l is not None:
            index &= self.ob_wd_grade == l
        ob_wd = self.ob_wd[index]
        pr_wd = self.pr_wd[index]
        delta = pr_wd - ob_wd
        delta = delta[ob_wd != 999017]
        delta[delta > 180] -= 360
        delta[delta < -180] += 360
        return float(np.mean(np.abs(delta)))

    def get_wd_rmse(self, l: typing.Optional[int] = None) -> float:
        assert l in (None, 0, 1, 2, 3, 4, 5, 6, 7)
        index = (~np.isnan(self.ob_wd)) & (~np.isnan(self.pr_wd))
        if l is not None:
            index &= self.ob_wd_grade == l
        ob_wd = self.ob_wd[index]
        pr_wd = self.pr_wd[index]
        delta = pr_wd - ob_wd
        delta = delta[ob_wd != 999017]
        delta[delta > 180] -= 360
        delta[delta < -180] += 360
        return float(np.mean(delta ** 2) ** 0.5)

    def get_wd_cs(self, l: typing.Optional[int] = None) -> float:
        assert l in (None, 0, 1, 2, 3, 4, 5, 6, 7)
        index = (~np.isnan(self.ob_wd)) & (~np.isnan(self.pr_wd))
        if l is not None:
            index &= self.ob_wd_grade == l
        ob_wd = self.ob_wd[index]
        pr_wd = self.pr_wd[index]
        delta = pr_wd - ob_wd
        delta = delta[ob_wd != 999017]
        delta[delta > 180] -= 360
        delta[delta < -180] += 360
        return float(np.mean(np.cos(delta)))

    def get_wd_cm(self) -> np.ndarray:
        return self.wd_cm

    def get_wd_oa(self, is_ws: bool = False) -> float:
        if is_ws:
            oa = np.zeros(self.n_grades - 1, dtype=np.float32) + np.nan
            for i in range(1, self.n_grades):
                wd_cm = np.zeros((8, 8), dtype=np.int_)
                for j in range(8):
                    for k in range(8):
                        wd_cm[j, k] = np.sum((self.ob_ws_grade == i) & (self.ob_wd_grade == j)
                                             & (self.pr_wd_grade == k))
                oa[i - 1] = np.sum(np.diag(wd_cm)) / np.sum(wd_cm)
        else:
            oa = np.sum(np.diag(self.wd_cm)) / self.wd_n
        return oa

    def get_wd_kappa(self, is_ws: bool = False) -> float:
        if is_ws:
            kappa0 = np.zeros(self.n_grades - 1, dtype=np.float32) + np.nan
            for i in range(1, self.n_grades):
                wd_cm = np.zeros((8, 8), dtype=np.int_)
                for j in range(8):
                    for k in range(8):
                        wd_cm[j, k] = np.sum((self.ob_ws_grade == i) & (self.ob_wd_grade == j)
                                             & (self.pr_wd_grade == k))
                a = np.sum(wd_cm, axis=1)
                a = a.astype(np.float32)
                b = np.sum(wd_cm, axis=0)
                b = b.astype(np.float32)
                pe = np.sum(a * b) / np.sum(wd_cm) / np.sum(wd_cm)
                oa = np.sum(np.diag(wd_cm)) / np.sum(wd_cm)
                kappa0[i - 1] = (oa - pe) / (1 - pe)
        else:
            a = np.sum(self.wd_cm, axis=1)
            a = a.astype(np.float32)
            b = np.sum(self.wd_cm, axis=0)
            b = b.astype(np.float32)
            pe = np.sum(a * b) / self.wd_n / self.wd_n
            kappa0 = (self.get_wd_oa() - pe) / (1 - pe)
        return kappa0

    def get_ws_me(self) -> float:
        index = (~np.isnan(self.ob_ws)) & (~np.isnan(self.pr_ws))
        ob_ws = self.ob_ws[index]
        pr_ws = self.pr_ws[index]
        return float(np.mean(pr_ws - ob_ws))

    def get_ws_mae(self, l: typing.Optional[int] = None) -> float:
        assert l in (None, 0, 1, 2, 3, 4, 5, 6, 7)
        index = (~np.isnan(self.ob_ws)) & (~np.isnan(self.pr_ws))
        if l is not None:
            index &= self.ob_wd_grade == l
        ob_ws = self.ob_ws[index]
        pr_ws = self.pr_ws[index]
        return float(np.mean(np.abs(pr_ws - ob_ws)))

    def get_ws_rmse(self, l: typing.Optional[int] = None) -> float:
        assert l in (None, 0, 1, 2, 3, 4, 5, 6, 7)
        index = (~np.isnan(self.ob_ws)) & (~np.isnan(self.pr_ws))
        if l is not None:
            index &= self.ob_wd_grade == l
        ob_ws = self.ob_ws[index]
        pr_ws = self.pr_ws[index]
        return float(np.mean((pr_ws - ob_ws) ** 2) ** 0.5)

    def get_ws_mre(self, l: typing.Optional[int] = None) -> float:
        assert l in (None, 0, 1, 2, 3, 4, 5, 6, 7)
        index = self.pr_ws + self.ob_ws > 0
        if l is not None:
            index &= self.ob_wd_grade == l
        ob_ws = self.ob_ws[index]
        pr_ws = self.pr_ws[index]
        return float(np.mean(np.abs((pr_ws - ob_ws) / (pr_ws + ob_ws))))

    def get_ws_r(self, l: typing.Optional[int] = None) -> float:
        assert l in (None, 0, 1, 2, 3, 4, 5, 6, 7)
        index = (~np.isnan(self.ob_ws)) & (~np.isnan(self.pr_ws))
        if l is not None:
            index &= self.ob_wd_grade == l
        ob_ws = self.ob_ws[index]
        pr_ws = self.pr_ws[index]
        return float(np.corrcoef(ob_ws, pr_ws)[0, 1])

    def get_ws_ts(self) -> np.ndarray:
        ts = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum((self.pr_ws_grade == i) & (self.ob_ws_grade == i))
            nb = np.sum((self.pr_ws_grade == i) & (self.ob_ws_grade != i))
            nc = np.sum((self.pr_ws_grade != i) & (self.ob_ws_grade == i))
            ts[i] = na / (na + nb + nc) if na + nb + nc != 0 else np.nan
        return ts

    def get_ws_ts2(self) -> np.ndarray:
        ts = np.zeros(self.n_grades - 1, dtype=np.float32)
        for i in range(self.n_grades - 1):
            na = np.sum((self.pr_ws_grade > i) & (self.ob_ws_grade > i))
            nb = np.sum((self.pr_ws_grade > i) & (self.ob_ws_grade <= i))
            nc = np.sum((self.pr_ws_grade <= i) & (self.ob_ws_grade > i))
            ts[i] = na / (na + nb + nc) if na + nb + nc != 0 else np.nan
        return ts

    def get_ws_ets(self) -> np.ndarray:
        ets = np.zeros(self.n_grades , dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum((self.pr_ws_grade == i) & (self.ob_ws_grade == i))
            nb = np.sum((self.pr_ws_grade == i) & (self.ob_ws_grade != i))
            nc = np.sum((self.pr_ws_grade != i) & (self.ob_ws_grade == i))
            nd = np.sum((self.pr_ws_grade != i) & (self.ob_ws_grade != i))
            r = (na + nb) / (na + nb + nc + nd) * (na + nc)
            ets[i] = (na - r) / (na + nb + nc - r) if na + nb + nc != 0 else np.nan
        return ets

    def get_ws_ets2(self) -> np.ndarray:
        ets = np.zeros(self.n_grades - 1, dtype=np.float32)
        for i in range(self.n_grades - 1):
            na = np.sum((self.pr_ws_grade > i) & (self.ob_ws_grade > i))
            nb = np.sum((self.pr_ws_grade > i) & (self.ob_ws_grade <= i))
            nc = np.sum((self.pr_ws_grade <= i) & (self.ob_ws_grade > i))
            nd = np.sum((self.pr_ws_grade <= i) & (self.ob_ws_grade <= i))
            r = (na + nb) / (na + nb + nc + nd) * (na + nc)
            ets[i] = (na - r) / (na + nb + nc - r) if na + nb + nc != 0 else np.nan
        return ets

    def get_ws_bias(self) -> np.ndarray:
        bias = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum((self.pr_ws_grade == i) & (self.ob_ws_grade == i))
            nb = np.sum((self.pr_ws_grade == i) & (self.ob_ws_grade != i))
            nc = np.sum((self.pr_ws_grade != i) & (self.ob_ws_grade == i))
            bias[i] = (na + nb) / (na + nc) if na + nc != 0 else np.nan
        return bias

    def get_ws_far(self) -> np.ndarray:
        far = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum((self.pr_ws_grade == i) & (self.ob_ws_grade == i))
            nb = np.sum((self.pr_ws_grade == i) & (self.ob_ws_grade != i))
            far[i] = nb / (na + nb) if na + nb != 0 else np.nan
        return far

    def get_ws_mar(self) -> np.ndarray:
        mar = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum((self.pr_ws_grade == i + 1) & (self.ob_ws_grade == i + 1))
            nc = np.sum((self.pr_ws_grade != i + 1) & (self.ob_ws_grade == i + 1))
            mar[i] = nc / (na + nc) if na + nc != 0 else np.nan
        return mar

    def get_ws_cm(self) -> np.ndarray:
        return self.ws_cm

    def get_ws_oa(self) -> float:
        return np.sum(np.diag(self.ws_cm)) / self.ws_n

    def get_ws_kappa(self) -> float:
        a = np.sum(self.ws_cm, axis=1)
        a = a.astype(np.float32)
        b = np.sum(self.ws_cm, axis=0)
        b = b.astype(np.float32)
        pe = np.sum(a * b) / self.ws_n / self.ws_n
        return (self.get_ws_oa() - pe) / (1 - pe)


def interp(data: np.ndarray, is_0_84: bool = False) -> np.ndarray:
    if is_0_84:
        for i in range(24):
            data[:, 3 * i + 1, :, :] = data[:, 3 * i, :, :] * 2 / 3 + data[:, 3 * i + 3, :, :] / 3
            data[:, 3 * i + 2, :, :] = data[:, 3 * i, :, :] / 3 + data[:, 3 * i + 3, :, :] * 2 / 3
        data[:, 73, :, :] = data[:, 72, :, :] * 5 / 6 + data[:, 78, :, :] / 6
        data[:, 74, :, :] = data[:, 72, :, :] * 2 / 3 + data[:, 78, :, :] / 3
        data[:, 75, :, :] = data[:, 72, :, :] / 2 + data[:, 78, :, :] / 2
        data[:, 76, :, :] = data[:, 72, :, :] / 3 + data[:, 78, :, :] * 2 / 3
        data[:, 77, :, :] = data[:, 72, :, :] / 6 + data[:, 78, :, :] * 5 / 6
        data[:, 79, :, :] = data[:, 78, :, :] * 5 / 6 + data[:, 84, :, :] / 6
        data[:, 80, :, :] = data[:, 78, :, :] * 2 / 3 + data[:, 84, :, :] / 3
        data[:, 81, :, :] = data[:, 78, :, :] / 2 + data[:, 84, :, :] / 2
        data[:, 82, :, :] = data[:, 78, :, :] / 3 + data[:, 84, :, :] * 2 / 3
        data[:, 83, :, :] = data[:, 78, :, :] / 6 + data[:, 84, :, :] * 5 / 6
    else:
        for i in range(8):
            data[:, 3 * i + 1, :, :] = data[:, 3 * i, :, :] * 2 / 3 + data[:, 3 * i + 3, :, :] / 3
            data[:, 3 * i + 2, :, :] = data[:, 3 * i, :, :] / 3 + data[:, 3 * i + 3, :, :] * 2 / 3
    return data


def format_time(second: float, is_abbreviation: bool = False) -> str:
    r"""Format time.

    :param second: A float number representing the number of seconds.
    :param is_abbreviation: A boolean variable representing whether processing to abbreviation.
        The default value is False.
    :return: A sequence of strings representing the time. For example: '43.5 seconds'
    :raise ValueError: The value of input parameter 'second' is wrong.
    """
    if second < 0:
        raise ValueError('The input parameter "second" cannot be negative.')
    elif is_abbreviation:
        if second <= 60:
            time_str = str(second) + 's'
        elif second <= 3600:
            time_str = str(second / 60) + 'm'
        else:
            time_str = str(second / 3600) + 'h'
    else:
        if second <= 1:
            time_str = str(second) + ' second'
        elif second <= 60:
            time_str = str(second) + ' seconds'
        elif second <= 3600:
            time_str = str(second / 60) + 'minutes'
        else:
            time_str = str(second / 3600) + 'hours'

    return time_str


def main() -> None:
    # data = list()
    # for year in range(2017, 2023):
    #     data.append(np.load(fr'D:\data\wind\nwp{year}.npy'))
    # data = np.vstack(data)
    # data = interp(data[:, 12:37, :, :])[:, 1:, :, :]
    # print(data.shape)
    # data = np.mean(data, axis=2)
    # index = np.ones((4382, 24), dtype=np.bool_)
    # for i in range(30):
    #     index &= ~np.isnan(data[..., i])
    # data = data[index, :]
    # print(data.shape)
    # mi = np.zeros((30, 30), dtype=np.float32) + np.nan
    # for i in range(30):
    #     mi[:, i] = fs.mutual_info_regression(data[:, :], data[:, i])
    # mi = (mi + mi.T) / 2
    # print(mi)
    # np.save(r'D:\Project\wind\97-uv\mi.npy', mi)
    mi = np.load(r'D:\Project\wind\97-uv\mi.npy')
    sns.heatmap(mi, cmap='binary', vmin=0, vmax=10, linewidths=0.3)
    plt.xticks(np.arange(30), ELEMENT_NAME_LIST, rotation=45, fontsize=7)
    plt.yticks(np.arange(30) + 0.5, ELEMENT_NAME_LIST, rotation=0, fontsize=7)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.savefig(r'D:\Project\wind\97-uv\图\mi.jpg', bbox_inches='tight', dpi=800)
    plt.cla()
    plt.close()

    ob = list()
    ec = list()
    pdfm = list()
    mlr = list()
    lgb = list()
    for year in range(2017, 2023):
        ob.append(np.load(fr'D:\Project\wind\97\ob_1_{year}.npy'))
        ec.append(np.load(fr'D:\Project\wind\97\nwp_1_{year}.npy'))
        pdfm.append(np.load(fr'D:\Project\wind\97\pdfm_4_{year}.npy'))
        mlr.append(np.load(fr'D:\Project\wind\97\mlr_4_{year}.npy'))
        lgb.append(np.load(fr'D:\Project\wind\97\lgb_3_{year}.npy'))
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
    for year in range(2017, 2023):
        ob2.append(np.load(fr'D:\Project\wind\97-uv\ob_{year}.npy'))
        ec2.append(np.load(fr'D:\Project\wind\97-uv\nwp_{year}.npy'))
        pdfm2.append(np.load(fr'D:\Project\wind\97-uv\pdfm_3_{year}.npy'))
        mlr2.append(np.load(fr'D:\Project\wind\97-uv\mlr_3_{year}.npy'))
        lgb2.append(np.load(fr'D:\Project\wind\97-uv\lgb_3_{year}.npy'))
    ob2 = np.vstack(ob2)
    ec2 = np.vstack(ec2)
    pdfm2 = np.vstack(pdfm2)
    mlr2 = np.vstack(mlr2)
    lgb2 = np.vstack(lgb2)
    ob2[:, :, :, 1] = ob
    ec2[:, :, :, 1] = ec
    pdfm2[:, :, :, 1] = pdfm
    mlr2[:, :, :, 1] = mlr
    lgb2[:, :, :, 1] = lgb

    ts_grade = np.zeros((2, 4, 8), dtype=np.float32) + np.nan
    wd_kappa_grade = np.zeros((4, 8), dtype=np.float32) + np.nan
    ws_rmse_wd = np.zeros((4, 9), dtype=np.float32) + np.nan
    wd_rmse_wd = np.zeros((4, 9), dtype=np.float32) + np.nan
    print('ECMWF-IFS')
    acc = Acc(ob2, ec2)
    print(acc.get_ws_r())
    print(acc.get_ws_mae())
    print(acc.get_ws_rmse())
    print(acc.get_ws_mre())
    print(acc.get_ws_oa())
    print(acc.get_ws_kappa())
    ts_grade[0, 0, :] = acc.get_ws_ts()[1:]
    ts_grade[1, 0, :] = acc.get_ws_ts2()
    wd_kappa_grade[0, :] = acc.get_wd_kappa(True)
    print(acc.get_wd_mae())
    print(acc.get_wd_rmse())
    print(acc.get_wd_oa())
    print(acc.get_wd_kappa())
    for i, j in enumerate([2, 1, 0, 7, 6, 5, 4, 3, 2]):
        wd_rmse_wd[0, i] = acc.get_wd_rmse(l=j)
        ws_rmse_wd[0, i] = acc.get_ws_rmse(l=j)
    print('PDFM')
    acc = Acc(ob2, pdfm2)
    print(acc.get_ws_r())
    print(acc.get_ws_mae())
    print(acc.get_ws_rmse())
    print(acc.get_ws_mre())
    print(acc.get_ws_oa())
    print(acc.get_ws_kappa())
    ts_grade[0, 1, :] = acc.get_ws_ts()[1:]
    ts_grade[1, 1, :] = acc.get_ws_ts2()
    wd_kappa_grade[1, :] = acc.get_wd_kappa(True)
    print(acc.get_wd_mae())
    print(acc.get_wd_rmse())
    print(acc.get_wd_oa())
    print(acc.get_wd_kappa())
    for i, j in enumerate([2, 1, 0, 7, 6, 5, 4, 3, 2]):
        wd_rmse_wd[1, i] = acc.get_wd_rmse(l=j)
        ws_rmse_wd[1, i] = acc.get_ws_rmse(l=j)
    print('MOS')
    acc = Acc(ob2, mlr2)
    print(acc.get_ws_r())
    print(acc.get_ws_mae())
    print(acc.get_ws_rmse())
    print(acc.get_ws_mre())
    print(acc.get_ws_oa())
    print(acc.get_ws_kappa())
    ts_grade[0, 2, :] = acc.get_ws_ts()[1:]
    ts_grade[1, 2, :] = acc.get_ws_ts2()
    wd_kappa_grade[2, :] = acc.get_wd_kappa(True)
    print(acc.get_wd_mae())
    print(acc.get_wd_rmse())
    print(acc.get_wd_oa())
    print(acc.get_wd_kappa())
    for i, j in enumerate([2, 1, 0, 7, 6, 5, 4, 3, 2]):
        wd_rmse_wd[2, i] = acc.get_wd_rmse(l=j)
        ws_rmse_wd[2, i] = acc.get_ws_rmse(l=j)
    print('LightGBM')
    acc = Acc(ob2, lgb2)
    print(acc.get_ws_r())
    print(acc.get_ws_mae())
    print(acc.get_ws_rmse())
    print(acc.get_ws_mre())
    print(acc.get_ws_oa())
    print(acc.get_ws_kappa())
    ts_grade[0, 3, :] = acc.get_ws_ts()[1:]
    ts_grade[1, 3, :] = acc.get_ws_ts2()
    wd_kappa_grade[3, :] = acc.get_wd_kappa(True)
    acc = Acc(ob2, lgb2)
    print(acc.get_wd_mae())
    print(acc.get_wd_rmse())
    print(acc.get_wd_oa())
    print(acc.get_wd_kappa())
    for i, j in enumerate([2, 1, 0, 7, 6, 5, 4, 3, 2]):
        wd_rmse_wd[3, i] = acc.get_wd_rmse(l=j)
        ws_rmse_wd[3, i] = acc.get_ws_rmse(l=j)

    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(8) - 0.3,
        height=ts_grade[0, 0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(8) - 0.1,
        height=ts_grade[0, 1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(8) + 0.1,
        height=ts_grade[0, 2, :],
        width=0.2,
        color='yellow',
        label='MOS'
    )
    plt.bar(
        x=np.arange(8) + 0.3,
        height=ts_grade[0, 3, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 8))
    plt.xticks(range(8), ['1级', '2级', '3级', '4级', '5级', '6级', '7级', '8级及以上'])
    plt.ylim((0, 1))
    plt.yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
    plt.xlabel('风速等级')
    plt.ylabel('TS')
    plt.legend(loc='upper right')
    plt.savefig(r'D:\Project\wind\97-uv\图\ts_grade.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(8) - 0.3,
        height=ts_grade[1, 0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(8) - 0.1,
        height=ts_grade[1, 1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(8) + 0.1,
        height=ts_grade[1, 2, :],
        width=0.2,
        color='yellow',
        label='MOS'
    )
    plt.bar(
        x=np.arange(8) + 0.3,
        height=ts_grade[1, 3, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 8))
    plt.xticks(range(8), ['≥0.3m/s', '≥1.6m/s', '≥3.4m/s', '≥5.5m/s', '≥8.0m/s', '≥10.8m/s', '≥13.9m/s', '≥17.2m/s'])
    plt.ylim((0, 1))
    plt.yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
    plt.xlabel('风速等级')
    plt.ylabel('TS')
    plt.legend(loc='upper right')
    plt.savefig(r'D:\Project\wind\97-uv\图\ts_grade+.jpg', bbox_inches='tight', dpi=800)
    plt.close()

    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(8) - 0.3,
        height=wd_kappa_grade[0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(8) - 0.1,
        height=wd_kappa_grade[1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(8) + 0.1,
        height=wd_kappa_grade[2, :],
        width=0.2,
        color='yellow',
        label='MOS'
    )
    plt.bar(
        x=np.arange(8) + 0.3,
        height=wd_kappa_grade[3, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 8))
    plt.xticks(range(8), ['1级', '2级', '3级', '4级', '5级', '6级', '7级', '8级及以上'])
    plt.ylim((0, 1))
    plt.yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
    plt.xlabel('风速等级')
    plt.ylabel('kappa')
    plt.legend(loc='upper left')
    plt.savefig(r'D:\Project\wind\97-uv\图\wd_kappa_grade.jpg', bbox_inches='tight', dpi=800)
    plt.close()

    n = round((arrow.get('2023') - arrow.get('2017')).total_seconds() / 3600 / 12)
    year_ind = np.zeros(n, np.int_) - 1
    month_ind = np.zeros(n, np.int_) - 1
    fhour_ind = np.zeros((n, 24), np.int_) - 1
    for i in range(n):
        time_arrow = arrow.get('2017').shift(hours=12 * i)
        year_ind[i] = time_arrow.datetime.year
        month_ind[i] = time_arrow.shift(hours=44).datetime.month
        for j in range(24):
            fhour_ind[i, j] = time_arrow.shift(hours=13 + j).datetime.hour

    ws_rmse_year = np.zeros((4, 6), dtype=np.float32) + np.nan
    wd_rmse_year = np.zeros((4, 6), dtype=np.float32) + np.nan
    for i in range(6):
        acc = Acc(ob2[year_ind == i + 2017, :, :], ec2[year_ind == i + 2017, :, :])
        ws_rmse_year[0, i] = acc.get_ws_rmse()
        wd_rmse_year[0, i] = acc.get_wd_rmse()
        acc = Acc(ob2[year_ind == i + 2017, :, :], pdfm2[year_ind == i + 2017, :, :])
        ws_rmse_year[1, i] = acc.get_ws_rmse()
        wd_rmse_year[1, i] = acc.get_wd_rmse()
        acc = Acc(ob2[year_ind == i + 2017, :, :], mlr2[year_ind == i + 2017, :, :])
        ws_rmse_year[2, i] = acc.get_ws_rmse()
        wd_rmse_year[2, i] = acc.get_wd_rmse()
        acc = Acc(ob2[year_ind == i + 2017, :, :], lgb2[year_ind == i + 2017, :, :])
        ws_rmse_year[3, i] = acc.get_ws_rmse()
        wd_rmse_year[3, i] = acc.get_wd_rmse()
    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(6) - 0.3,
        height=ws_rmse_year[0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(6) - 0.1,
        height=ws_rmse_year[1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(6) + 0.1,
        height=ws_rmse_year[2, :],
        width=0.2,
        color='yellow',
        label='MOS'
    )
    plt.bar(
        x=np.arange(6) + 0.3,
        height=ws_rmse_year[3, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 6))
    plt.xticks(range(6), [f'{x + 2017}年' for x in range(6)])
    plt.ylim((0, 1.6))
    plt.yticks((0, 0.4, 0.8, 1.2, 1.6,))
    plt.xlabel('年份')
    plt.ylabel('风速RMSE（m/s）')
    plt.legend(loc='upper center', ncols=4)
    plt.savefig(r'D:\Project\wind\97-uv\图\ws_rmse_year.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(6) - 0.3,
        height=wd_rmse_year[0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(6) - 0.1,
        height=wd_rmse_year[1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(6) + 0.1,
        height=wd_rmse_year[2, :],
        width=0.2,
        color='yellow',
        label='MOS'
    )
    plt.bar(
        x=np.arange(6) + 0.3,
        height=wd_rmse_year[3, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 6))
    plt.xticks(range(6), [f'{x + 2017}年' for x in range(6)])
    plt.ylim((0, 100))
    plt.yticks((0, 20, 40, 60, 80, 100))
    plt.xlabel('年份')
    plt.ylabel('风向RMSE（°）')
    plt.legend(loc='upper center', ncols=4)
    plt.savefig(r'D:\Project\wind\97-uv\图\wd_rmse_year.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    print(np.nanmin(ws_rmse_year), np.nanmax(ws_rmse_year))
    print(np.nanmin(wd_rmse_year), np.nanmax(wd_rmse_year))
    ws_rmse_month = np.zeros((4, 12), dtype=np.float32) + np.nan
    wd_rmse_month = np.zeros((4, 12), dtype=np.float32) + np.nan
    for i in range(12):
        acc = Acc(ob2[month_ind == i + 1, :, :], ec2[month_ind == i + 1, :, :])
        ws_rmse_month[0, i] = acc.get_ws_rmse()
        wd_rmse_month[0, i] = acc.get_wd_rmse()
        acc = Acc(ob2[month_ind == i + 1, :, :], pdfm2[month_ind == i + 1, :, :])
        ws_rmse_month[1, i] = acc.get_ws_rmse()
        wd_rmse_month[1, i] = acc.get_wd_rmse()
        acc = Acc(ob2[month_ind == i + 1, :, :], mlr2[month_ind == i + 1, :, :])
        ws_rmse_month[2, i] = acc.get_ws_rmse()
        wd_rmse_month[2, i] = acc.get_wd_rmse()
        acc = Acc(ob2[month_ind == i + 1, :, :], lgb2[month_ind == i + 1, :, :])
        ws_rmse_month[3, i] = acc.get_ws_rmse()
        wd_rmse_month[3, i] = acc.get_wd_rmse()
    print('month, ws', np.argmin(ws_rmse_month[3, :]), np.min(ws_rmse_month[3, :]))
    print('month, ws', np.argmax(ws_rmse_month[3, :]), np.max(ws_rmse_month[3, :]))
    print('month, wd', np.argmin(wd_rmse_month[3, :]), np.min(wd_rmse_month[3, :]))
    print('month, wd', np.argmax(wd_rmse_month[3, :]), np.max(wd_rmse_month[3, :]))
    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(12) - 0.3,
        height=ws_rmse_month[0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(12) - 0.1,
        height=ws_rmse_month[1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(12) + 0.1,
        height=ws_rmse_month[2, :],
        width=0.2,
        color='yellow',
        label='MOS'
    )
    plt.bar(
        x=np.arange(12) + 0.3,
        height=ws_rmse_month[3, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 12))
    plt.xticks(range(12), [f'{x + 1}月' for x in range(12)])
    plt.ylim((0, 1.6))
    plt.yticks((0, 0.4, 0.8, 1.2, 1.6,))
    plt.xlabel('月份')
    plt.ylabel('风速RMSE（m/s）')
    plt.legend(loc='upper center', ncols=4)
    plt.savefig(r'D:\Project\wind\97-uv\图\ws_rmse_month.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(12) - 0.3,
        height=wd_rmse_month[0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(12) - 0.1,
        height=wd_rmse_month[1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(12) + 0.1,
        height=wd_rmse_month[2, :],
        width=0.2,
        color='yellow',
        label='MOS'
    )
    plt.bar(
        x=np.arange(12) + 0.3,
        height=wd_rmse_month[3, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 12))
    plt.xticks(range(12), [f'{x + 1}月' for x in range(12)])
    plt.ylim((0, 100))
    plt.yticks((0, 20, 40, 60, 80, 100))
    plt.xlabel('月份')
    plt.ylabel('风向RMSE（°）')
    plt.legend(loc='upper center', ncols=4)
    plt.savefig(r'D:\Project\wind\97-uv\图\wd_rmse_month.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    print(np.nanmin(ws_rmse_month), np.nanmax(ws_rmse_month))
    print(np.nanmin(wd_rmse_month), np.nanmax(wd_rmse_month))
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
    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(24) - 0.3,
        height=ws_rmse_vt[0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(24) - 0.1,
        height=ws_rmse_vt[1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(24) + 0.1,
        height=ws_rmse_vt[2, :],
        width=0.2,
        color='yellow',
        label='MOS'
    )
    plt.bar(
        x=np.arange(24) + 0.3,
        height=ws_rmse_vt[3, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 24))
    plt.xticks(range(24), [str(x + 1) for x in range(24)])
    plt.ylim((0, 1.6))
    plt.yticks((0, 0.4, 0.8, 1.2, 1.6,))
    plt.xlabel('预报时效（h）')
    plt.ylabel('风速RMSE（m/s）')
    plt.legend(loc='upper center', ncols=4)
    plt.savefig(r'D:\Project\wind\97-uv\图\ws_rmse_vt.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(24) - 0.3,
        height=wd_rmse_vt[0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(24) - 0.1,
        height=wd_rmse_vt[1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(24) + 0.1,
        height=wd_rmse_vt[2, :],
        width=0.2,
        color='yellow',
        label='MOS'
    )
    plt.bar(
        x=np.arange(24) + 0.3,
        height=wd_rmse_vt[3, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 24))
    plt.xticks(range(24), [str(x + 1) for x in range(24)])
    plt.ylim((0, 100))
    plt.yticks((0, 20, 40, 60, 80, 100))
    plt.xlabel('预报时效（h）')
    plt.ylabel('风向RMSE（°）')
    plt.legend(loc='upper center', ncols=4)
    plt.savefig(r'D:\Project\wind\97-uv\图\wd_rmse_vt.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    print(np.nanmin(ws_rmse_vt), np.nanmax(ws_rmse_vt))
    print(np.nanmin(wd_rmse_vt), np.nanmax(wd_rmse_vt))
    ws_rmse_fhour = np.zeros((4, 24), dtype=np.float32) + np.nan
    wd_rmse_fhour = np.zeros((4, 24), dtype=np.float32) + np.nan
    for i in range(24):
        acc = Acc(ob2[fhour_ind == i, :], ec2[fhour_ind == i, :])
        ws_rmse_fhour[0, i] = acc.get_ws_rmse()
        wd_rmse_fhour[0, i] = acc.get_wd_rmse()
        acc = Acc(ob2[fhour_ind == i, :], pdfm2[fhour_ind == i, :])
        ws_rmse_fhour[1, i] = acc.get_ws_rmse()
        wd_rmse_fhour[1, i] = acc.get_wd_rmse()
        acc = Acc(ob2[fhour_ind == i, :], mlr2[fhour_ind == i, :])
        ws_rmse_fhour[2, i] = acc.get_ws_rmse()
        wd_rmse_fhour[2, i] = acc.get_wd_rmse()
        acc = Acc(ob2[fhour_ind == i, :], lgb2[fhour_ind == i, :])
        ws_rmse_fhour[3, i] = acc.get_ws_rmse()
        wd_rmse_fhour[3, i] = acc.get_wd_rmse()
    print('fhour, ws', np.argmin(ws_rmse_fhour[3, :]), np.min(ws_rmse_fhour[3, :]))
    print('fhour, ws', np.argmax(ws_rmse_fhour[3, :]), np.max(ws_rmse_fhour[3, :]))
    print('fhour, wd', np.argmin(wd_rmse_fhour[3, :]), np.min(wd_rmse_fhour[3, :]))
    print('fhour, wd', np.argmax(wd_rmse_fhour[3, :]), np.max(wd_rmse_fhour[3, :]))
    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(24) - 0.3,
        height=ws_rmse_fhour[0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(24) - 0.1,
        height=ws_rmse_fhour[1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(24) + 0.1,
        height=ws_rmse_fhour[2, :],
        width=0.2,
        color='yellow',
        label='MOS'
    )
    plt.bar(
        x=np.arange(24) + 0.3,
        height=ws_rmse_fhour[3, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 24))
    plt.xticks(range(24), [f'{x}:00' for x in range(24)])
    plt.ylim((0, 1.6))
    plt.yticks((0, 0.4, 0.8, 1.2, 1.6,))
    plt.xlabel('预报时间（UTC）')
    plt.ylabel('风速RMSE（m/s）')
    plt.legend(loc='upper center', ncols=4)
    plt.savefig(r'D:\Project\wind\97-uv\图\ws_rmse_fhour.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(24) - 0.3,
        height=wd_rmse_fhour[0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(24) - 0.1,
        height=wd_rmse_fhour[1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(24) + 0.1,
        height=wd_rmse_fhour[2, :],
        width=0.2,
        color='yellow',
        label='MOS'
    )
    plt.bar(
        x=np.arange(24) + 0.3,
        height=wd_rmse_fhour[3, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 24))
    plt.xticks(range(24), [f'{x}:00' for x in range(24)])
    plt.ylim((0, 100))
    plt.yticks((0, 20, 40, 60, 80, 100))
    plt.xlabel('预报时间（UTC）')
    plt.ylabel('风向RMSE（°）')
    plt.legend(loc='upper center', ncols=4)
    plt.savefig(r'D:\Project\wind\97-uv\图\wd_rmse_fhour.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    print(np.nanmin(ws_rmse_fhour), np.nanmax(ws_rmse_fhour))
    print(np.nanmin(wd_rmse_fhour), np.nanmax(wd_rmse_fhour))

    station_information = pd.read_csv(r'D:\Project\wind\国家气象观测站.csv', encoding='gb2312', low_memory=False)
    station_information.sort_values(by=['台站号'], inplace=True)
    station_information.reset_index(drop=True, inplace=True)
    sta = station_information.loc[:, ('台站号', '经度', '纬度')]
    sta = meb.sta_data(sta, columns=['id', 'lon', 'lat'])
    sta.loc[:, 'level'] = 0
    sta.loc[:, 'time'] = '2024-01-01 08:00:00'
    sta.loc[:, 'dtime'] = 0
    ws_rmse_sta = np.zeros((4, 97), dtype=np.float32) + np.nan
    wd_rmse_sta = np.zeros((4, 97), dtype=np.float32) + np.nan
    ws_rmse_improvement_sta = np.zeros((3, 97), dtype=np.float32) + np.nan
    wd_rmse_improvement_sta = np.zeros((3, 97), dtype=np.float32) + np.nan
    for i in range(97):
        acc = Acc(ob2[:, :, i], ec2[:, :, i])
        ws_rmse_sta[0, i] = acc.get_ws_rmse()
        wd_rmse_sta[0, i] = acc.get_wd_rmse()
        acc = Acc(ob2[:, :, i], pdfm2[:, :, i])
        ws_rmse_sta[1, i] = acc.get_ws_rmse()
        wd_rmse_sta[1, i] = acc.get_wd_rmse()
        ws_rmse_improvement_sta[0, i] = (ws_rmse_sta[0, i] - ws_rmse_sta[1, i]) / ws_rmse_sta[0, i] * 100
        wd_rmse_improvement_sta[0, i] = (wd_rmse_sta[0, i] - wd_rmse_sta[1, i]) / wd_rmse_sta[0, i] * 100
        acc = Acc(ob2[:, :, i], mlr2[:, :, i])
        ws_rmse_sta[2, i] = acc.get_ws_rmse()
        wd_rmse_sta[2, i] = acc.get_wd_rmse()
        ws_rmse_improvement_sta[1, i] = (ws_rmse_sta[0, i] - ws_rmse_sta[2, i]) / ws_rmse_sta[0, i] * 100
        wd_rmse_improvement_sta[1, i] = (wd_rmse_sta[0, i] - wd_rmse_sta[2, i]) / wd_rmse_sta[0, i] * 100
        acc = Acc(ob2[:, :, i], lgb2[:, :, i])
        ws_rmse_sta[3, i] = acc.get_ws_rmse()
        wd_rmse_sta[3, i] = acc.get_wd_rmse()
        ws_rmse_improvement_sta[2, i] = (ws_rmse_sta[0, i] - ws_rmse_sta[3, i]) / ws_rmse_sta[0, i] * 100
        wd_rmse_improvement_sta[2, i] = (wd_rmse_sta[0, i] - wd_rmse_sta[3, i]) / wd_rmse_sta[0, i] * 100
    # cmap, clevs = meb.def_cmap_clevs(
    #     meb.def_cmap_clevs(meb.cmaps.mae, vmin=-20, vmax=80)[0],
    #     clevs=[-20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80]
    # )
    # sta.loc[:, 'data0'] = ws_rmse_improvement_sta[0, :]
    # meb.tool.plot_tools.scatter_sta(
    #     sta0=sta,
    #     map_extend=[108.65, 114.4, 24.5, 30.25],
    #     clevs=clevs,
    #     cmap=cmap,
    #     title=[''],
    #     save_path=r'D:\Project\wind\97-uv\图\pdfm_ws_rmse_improvement_sta.jpg',
    #     dpi=800
    # )
    # sta.loc[:, 'data0'] = ws_rmse_improvement_sta[1, :]
    # meb.tool.plot_tools.scatter_sta(
    #     sta0=sta,
    #     map_extend=[108.65, 114.4, 24.5, 30.25],
    #     clevs=clevs,
    #     cmap=cmap,
    #     title=[''],
    #     save_path=r'D:\Project\wind\97-uv\图\mlr_ws_rmse_improvement_sta.jpg',
    #     dpi=800
    # )
    # sta.loc[:, 'data0'] = ws_rmse_improvement_sta[2, :]
    # meb.tool.plot_tools.scatter_sta(
    #     sta0=sta,
    #     map_extend=[108.65, 114.4, 24.5, 30.25],
    #     clevs=clevs,
    #     cmap=cmap,
    #     title=[''],
    #     save_path=r'D:\Project\wind\97-uv\图\lgb_ws_rmse_improvement_sta.jpg',
    #     dpi=800
    # )
    # print(np.nanmin(ws_rmse_improvement_sta, axis=1))
    # print(np.nanmean(ws_rmse_improvement_sta, axis=1))
    # print(np.nanmedian(ws_rmse_improvement_sta, axis=1))
    # print(np.nanmax(ws_rmse_improvement_sta, axis=1))
    # cmap, clevs = meb.def_cmap_clevs(
    #     meb.def_cmap_clevs(meb.cmaps.mae, vmin=-20, vmax=80)[0],
    #     clevs=[-20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80]
    # )
    # sta.loc[:, 'data0'] = wd_rmse_improvement_sta[0, :]
    # meb.tool.plot_tools.scatter_sta(
    #     sta0=sta,
    #     map_extend=[108.65, 114.4, 24.5, 30.25],
    #     clevs=clevs,
    #     cmap=cmap,
    #     title=[''],
    #     save_path=r'D:\Project\wind\97-uv\图\pdfm_wd_rmse_improvement_sta.jpg',
    #     dpi=800
    # )
    # sta.loc[:, 'data0'] = wd_rmse_improvement_sta[1, :]
    # meb.tool.plot_tools.scatter_sta(
    #     sta0=sta,
    #     map_extend=[108.65, 114.4, 24.5, 30.25],
    #     clevs=clevs,
    #     cmap=cmap,
    #     title=[''],
    #     save_path=r'D:\Project\wind\97-uv\图\mlr_wd_rmse_improvement_sta.jpg',
    #     dpi=800
    # )
    # sta.loc[:, 'data0'] = wd_rmse_improvement_sta[2, :]
    # meb.tool.plot_tools.scatter_sta(
    #     sta0=sta,
    #     map_extend=[108.65, 114.4, 24.5, 30.25],
    #     clevs=clevs,
    #     cmap=cmap,
    #     title=[''],
    #     save_path=r'D:\Project\wind\97-uv\图\lgb_wd_rmse_improvement_sta.jpg',
    #     dpi=800
    # )
    # print(np.nanmin(wd_rmse_improvement_sta, axis=1))
    # print(np.nanmean(wd_rmse_improvement_sta, axis=1))
    # print(np.nanmedian(wd_rmse_improvement_sta, axis=1))
    # print(np.nanmax(wd_rmse_improvement_sta, axis=1))

    # plt.figure(figsize=(5, 5))
    # plt.scatter(
    #     x=station_information.loc[:, '经度'],
    #     y=rmse_improvement_sta[0, :],
    #     s=0.2,
    #     c='green',
    #     label='PDFM'
    # )
    # plt.scatter(
    #     x=station_information.loc[:, '经度'],
    #     y=rmse_improvement_sta[1, :],
    #     s=0.2,
    #     c='yellow',
    #     label='MOS'
    # )
    # plt.scatter(
    #     x=station_information.loc[:, '经度'],
    #     y=rmse_improvement_sta[2, :],
    #     s=0.2,
    #     c='red',
    #     label='LightGBM'
    # )
    # plt.xlim((108.65, 114.4))
    # plt.xticks([109, 110, 111, 112, 113, 114], ['109°', '110°', '111°', '112°', '113°', '114°E'])
    # plt.ylim((-20, 80))
    # plt.yticks((-20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80))
    # plt.xlabel('经度')
    # plt.ylabel('RMSE改善率/m')
    # plt.legend()
    # plt.savefig(r'D:\Project\wind\97-uv\图\rmse_improvement-lon.jpg', bbox_inches='tight', dpi=800)
    # plt.close()
    # print('经度')
    # print(np.corrcoef(station_information.loc[:, '经度'], rmse_improvement_sta[0, :])[0, 1])
    # print(np.corrcoef(station_information.loc[:, '经度'], rmse_improvement_sta[1, :])[0, 1])
    # print(np.corrcoef(station_information.loc[:, '经度'], rmse_improvement_sta[2, :])[0, 1])
    # plt.figure(figsize=(5, 5))
    # plt.scatter(
    #     x=station_information.loc[:, '纬度'],
    #     y=rmse_improvement_sta[0, :],
    #     s=0.2,
    #     c='green',
    #     label='PDFM'
    # )
    # plt.scatter(
    #     x=station_information.loc[:, '纬度'],
    #     y=rmse_improvement_sta[1, :],
    #     s=0.2,
    #     c='yellow',
    #     label='MOS'
    # )
    # plt.scatter(
    #     x=station_information.loc[:, '纬度'],
    #     y=rmse_improvement_sta[2, :],
    #     s=0.2,
    #     c='red',
    #     label='LightGBM'
    # )
    # plt.xlim((24.5, 30.25))
    # plt.xticks([25, 26, 27, 28, 29, 30], ['25°', '26°', '27°', '28°', '29°', '30°N'])
    # plt.ylim((-20, 80))
    # plt.yticks((-20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80))
    # plt.xlabel('纬度')
    # plt.ylabel('RMSE改善率/m')
    # plt.legend()
    # plt.savefig(r'D:\Project\wind\97-uv\图\rmse_improvement-lat.jpg', bbox_inches='tight', dpi=800)
    # plt.close()
    # print('纬度')
    # print(np.corrcoef(station_information.loc[:, '纬度'], rmse_improvement_sta[0, :])[0, 1])
    # print(np.corrcoef(station_information.loc[:, '纬度'], rmse_improvement_sta[1, :])[0, 1])
    # print(np.corrcoef(station_information.loc[:, '纬度'], rmse_improvement_sta[2, :])[0, 1])
    # plt.figure(figsize=(5, 5))
    # plt.scatter(
    #     x=station_information.loc[:, '海拔'],
    #     y=rmse_improvement_sta[0, :],
    #     s=0.2,
    #     c='green',
    #     label='PDFM'
    # )
    # plt.scatter(
    #     x=station_information.loc[:, '海拔'],
    #     y=rmse_improvement_sta[1, :],
    #     s=0.2,
    #     c='yellow',
    #     label='MOS'
    # )
    # plt.scatter(
    #     x=station_information.loc[:, '海拔'],
    #     y=rmse_improvement_sta[2, :],
    #     s=0.2,
    #     c='red',
    #     label='LightGBM'
    # )
    # plt.xlim((0, 1500))
    # plt.xticks([0, 300, 600, 900, 1200, 1500], ['0', '300', '600', '900', '1200', '1500'])
    # plt.ylim((-20, 80))
    # plt.yticks((-20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80))
    # plt.xlabel('高程/m')
    # plt.ylabel('RMSE改善率/m')
    # plt.legend()
    # plt.savefig(r'D:\Project\wind\97-uv\图\rmse_improvement-alti.jpg', bbox_inches='tight', dpi=800)
    # plt.close()
    # print('高程')
    # print(np.corrcoef(station_information.loc[:, '海拔'], rmse_improvement_sta[0, :])[0, 1])
    # print(np.corrcoef(station_information.loc[:, '海拔'], rmse_improvement_sta[1, :])[0, 1])
    # print(np.corrcoef(station_information.loc[:, '海拔'], rmse_improvement_sta[2, :])[0, 1])

    plt.figure(figsize=(5, 5))
    sns.violinplot(
        data={
            'ECMWF-IFS': ws_rmse_sta[0, :],
            'PDFM': ws_rmse_sta[1, :],
            'MOS': ws_rmse_sta[2, :],
            'LightGBM': ws_rmse_sta[3, :]
        },
        palette=['blue', 'green', 'yellow', 'red']
    )
    plt.xlabel('模型')
    plt.ylabel('风速RMSE（m/s）')
    plt.savefig(r'D:\Project\wind\97-uv\图\ws_rmse_boxplot.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    print('ws_rmse')
    print(np.median(ws_rmse_sta, axis=1))
    plt.figure(figsize=(5, 5))
    sns.violinplot(
        data={
            'ECMWF-IFS': wd_rmse_sta[0, :],
            'PDFM': wd_rmse_sta[0, :],
            'MOS': wd_rmse_sta[1, :],
            'LightGBM': wd_rmse_sta[2, :]
        },
        palette=['blue', 'green', 'yellow', 'red']
    )
    plt.xlabel('模型')
    plt.ylabel('风向RMSE（°）')
    plt.savefig(r'D:\Project\wind\97-uv\图\wd_rmse_boxplot.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    print('wd_rmse')
    print(np.median(wd_rmse_sta, axis=1))
    plt.figure(figsize=(5, 5))
    sns.violinplot(
        data={
            'PDFM': ws_rmse_improvement_sta[0, :],
            'MOS': ws_rmse_improvement_sta[1, :],
            'LightGBM': ws_rmse_improvement_sta[2, :]
        },
        palette=['green', 'yellow', 'red']
    )
    plt.xlabel('模型')
    plt.ylabel('风速RMSE改善率（%）')
    plt.savefig(r'D:\Project\wind\97-uv\图\ws_rmse_improvement_boxplot.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    print('ws_rmse_improvement')
    print(np.median(ws_rmse_improvement_sta, axis=1))
    plt.figure(figsize=(5, 5))
    sns.violinplot(
        data={
            'PDFM': wd_rmse_improvement_sta[0, :],
            'MOS': wd_rmse_improvement_sta[1, :],
            'LightGBM': wd_rmse_improvement_sta[2, :]
        },
        palette=['green', 'yellow', 'red']
    )
    plt.xlabel('模型')
    plt.ylabel('风向RMSE改善率（%）')
    plt.savefig(r'D:\Project\wind\97-uv\图\wd_rmse_improvement_boxplot.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    print('wd_rmse_improvement')
    print(np.median(wd_rmse_improvement_sta, axis=1))

    ws_rmse_improvement_wd = np.zeros((3, 9), dtype=np.float32) + np.nan
    wd_rmse_improvement_wd = np.zeros((3, 9), dtype=np.float32) + np.nan
    ws_rmse_improvement_wd[0, :] = (ws_rmse_wd[0, :] - ws_rmse_wd[1, :]) / ws_rmse_wd[0, :] * 100
    ws_rmse_improvement_wd[1, :] = (ws_rmse_wd[0, :] - ws_rmse_wd[2, :]) / ws_rmse_wd[0, :] * 100
    ws_rmse_improvement_wd[2, :] = (ws_rmse_wd[0, :] - ws_rmse_wd[3, :]) / ws_rmse_wd[0, :] * 100
    wd_rmse_improvement_wd[0, :] = (wd_rmse_wd[0, :] - wd_rmse_wd[1, :]) / wd_rmse_wd[0, :] * 100
    wd_rmse_improvement_wd[1, :] = (wd_rmse_wd[0, :] - wd_rmse_wd[2, :]) / wd_rmse_wd[0, :] * 100
    wd_rmse_improvement_wd[2, :] = (wd_rmse_wd[0, :] - wd_rmse_wd[3, :]) / wd_rmse_wd[0, :] * 100
    labels = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
    angles = np.linspace(0, 2 * np.pi, 9).tolist()
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, ws_rmse_wd[0, :], color='blue', label='ECMWF-IFS')
    ax.plot(angles, ws_rmse_wd[1, :], color='green', label='PDFM')
    ax.plot(angles, ws_rmse_wd[2, :], color='yellow', label='MOS')
    ax.plot(angles, ws_rmse_wd[3, :], color='red', label='LightGBM')
    ax.set_xticks(angles[:-1], labels)
    ax.set_rgrids((0, 0.5, 1, 1.5, 2), angle=90)
    # ax.set_ylabel('RMSE（m/s）')
    plt.savefig(r'D:\Project\wind\97-uv\图\radar_ws_rmse.jpg', bbox_inches='tight', dpi=800)
    print(np.nanmax(ws_rmse_wd))
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, wd_rmse_wd[0, :], color='blue', label='ECMWF-IFS')
    ax.plot(angles, wd_rmse_wd[1, :], color='green', label='PDFM')
    ax.plot(angles, wd_rmse_wd[2, :], color='yellow', label='MOS')
    ax.plot(angles, wd_rmse_wd[3, :], color='red', label='LightGBM')
    ax.set_xticks(angles[:-1], labels)
    ax.set_rgrids((0, 25, 50, 75, 100), angle=90)
    # ax.set_ylabel('RMSE（°）')
    plt.savefig(r'D:\Project\wind\97-uv\图\radar_wd_rmse.jpg', bbox_inches='tight', dpi=800)
    print(np.nanmax(wd_rmse_wd))
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, ws_rmse_improvement_wd[0, :], color='green', label='PDFM')
    ax.plot(angles, ws_rmse_improvement_wd[1, :], color='yellow', label='MOS')
    ax.plot(angles, ws_rmse_improvement_wd[2, :], color='red', label='LightGBM')
    ax.set_xticks(angles[:-1], labels)
    ax.set_rgrids((0, 10, 20, 30, 40, 50), angle=90)
    # ax.set_ylabel('RMSE（m/s）')
    plt.savefig(r'D:\Project\wind\97-uv\图\radar_ws_improvement_rmse.jpg', bbox_inches='tight', dpi=800)
    print(np.nanmin(ws_rmse_improvement_wd), np.nanmax(ws_rmse_improvement_wd))
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, wd_rmse_improvement_wd[0, :], color='green', label='PDFM')
    ax.plot(angles, wd_rmse_improvement_wd[1, :], color='yellow', label='MOS')
    ax.plot(angles, wd_rmse_improvement_wd[2, :], color='red', label='LightGBM')
    ax.set_xticks(angles[:-1], labels)
    ax.set_rgrids((-10, -5, 0, 5, 10, 15), angle=90)
    # ax.set_ylabel('RMSE（°）')
    plt.savefig(r'D:\Project\wind\97-uv\图\radar_wd_improvement_rmse.jpg', bbox_inches='tight', dpi=800)
    print(np.nanmin(wd_rmse_improvement_wd), np.nanmax(wd_rmse_improvement_wd))

    lgb11 = list()
    lgb22 = list()
    lgb33 = list()
    lgb44 = list()
    for year in range(2017, 2023):
        lgb11.append(np.load(fr'D:\Project\wind\97\lgb_1_{year}.npy'))
        lgb22.append(np.load(fr'D:\Project\wind\97\lgb_2_{year}.npy'))
        lgb33.append(np.load(fr'D:\Project\wind\97\lgb_3_{year}.npy'))
        lgb44.append(np.load(fr'D:\Project\wind\97\lgb_4_{year}.npy'))
    lgb11 = np.vstack(lgb11)
    lgb22 = np.vstack(lgb22)
    lgb33 = np.vstack(lgb33)
    lgb44 = np.vstack(lgb44)
    lgb_1 = list()
    lgb_2 = list()
    lgb_3 = list()
    lgb_4 = list()
    for year in range(2017, 2023):
        lgb_1.append(np.load(fr'D:\Project\wind\97-uv\lgb_1_{year}.npy'))
        lgb_2.append(np.load(fr'D:\Project\wind\97-uv\lgb_2_{year}.npy'))
        lgb_3.append(np.load(fr'D:\Project\wind\97-uv\lgb_3_{year}.npy'))
        lgb_4.append(np.load(fr'D:\Project\wind\97-uv\lgb_4_{year}.npy'))
    lgb_1 = np.vstack(lgb_1)
    lgb_2 = np.vstack(lgb_2)
    lgb_3 = np.vstack(lgb_3)
    lgb_4 = np.vstack(lgb_4)
    lgb_1[:, :, :, 1] = lgb11
    lgb_2[:, :, :, 1] = lgb22
    lgb_3[:, :, :, 1] = lgb33
    lgb_4[:, :, :, 1] = lgb44
    print('LightGBM1')
    acc = Acc(ob2, lgb_1)
    print(acc.get_ws_r())
    print(acc.get_ws_mae())
    print(acc.get_ws_rmse())
    print(acc.get_ws_mre())
    print(acc.get_ws_oa())
    print(acc.get_ws_kappa())
    print(acc.get_wd_mae())
    print(acc.get_wd_rmse())
    print(acc.get_wd_oa())
    print(acc.get_wd_kappa())
    print('LightGBM2')
    acc = Acc(ob2, lgb_2)
    print(acc.get_ws_r())
    print(acc.get_ws_mae())
    print(acc.get_ws_rmse())
    print(acc.get_ws_mre())
    print(acc.get_ws_oa())
    print(acc.get_ws_kappa())
    print(acc.get_wd_mae())
    print(acc.get_wd_rmse())
    print(acc.get_wd_oa())
    print(acc.get_wd_kappa())
    print('LightGBM3')
    acc = Acc(ob2, lgb_3)
    print(acc.get_ws_r())
    print(acc.get_ws_mae())
    print(acc.get_ws_rmse())
    print(acc.get_ws_mre())
    print(acc.get_ws_oa())
    print(acc.get_ws_kappa())
    print(acc.get_wd_mae())
    print(acc.get_wd_rmse())
    print(acc.get_wd_oa())
    print(acc.get_wd_kappa())
    print('LightGBM4')
    acc = Acc(ob2, lgb_4)
    print(acc.get_ws_r())
    print(acc.get_ws_mae())
    print(acc.get_ws_rmse())
    print(acc.get_ws_mre())
    print(acc.get_ws_oa())
    print(acc.get_ws_kappa())
    print(acc.get_wd_mae())
    print(acc.get_wd_rmse())
    print(acc.get_wd_oa())
    print(acc.get_wd_kappa())


if __name__ == '__main__':
    print('The program "access97_uv.py" is beginning.')
    start = arrow.now()

    main()

    end = arrow.now()
    running_time = (end - start).total_seconds()

    print('The program "access97_uv.py" runs out in {:s}.'.format(format_time(running_time)))
