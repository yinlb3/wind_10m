#!user/bin.python3

"""
Founded in 2024-07-27
Modified in 2024-10-13
@author: yinlb
"""
import os
import sys

import arrow
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from meteva import base as meb


THRES = (0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2)


class Acc:
    def __init__(self, ob: np.ndarray, pr: np.ndarray):
        self.thres = THRES
        self.n_grades = len(self.thres) + 1
        self.ob = ob
        self.pr = pr
        self.ob_grade = np.zeros_like(self.ob, dtype=np.int_) - 1
        self.ob_grade[~np.isnan(self.ob)] = 0
        for i in range(self.n_grades - 1):
            self.ob_grade[self.ob >= self.thres[i]] = i + 1
        self.pr_grade = np.zeros_like(self.pr, dtype=np.int_) - 1
        self.pr_grade[~np.isnan(self.pr)] = 0
        for i in range(self.n_grades - 1):
            self.pr_grade[self.pr >= self.thres[i]] = i + 1
        self.hxjz = np.zeros((self.n_grades + 1, self.n_grades + 1), dtype=np.int_)
        for i in range(self.n_grades + 1):
            for j in range(self.n_grades + 1):
                self.hxjz[i, j] = np.sum((self.ob_grade == i) & (self.pr_grade == j))
        self.n = np.sum(self.hxjz)

    def get_me(self) -> float:
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.mean(pr - ob))

    def get_mae(self) -> float:
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.mean(np.abs(pr - ob)))

    def get_rmse(self) -> float:
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.mean((pr - ob) ** 2) ** 0.5)

    def get_mre(self) -> float:
        index = self.pr + self.ob > 0
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.mean(np.abs((pr - ob) / (pr + ob))))

    def get_fs(self) -> float:
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob_grade[index]
        pr = self.pr_grade[index]
        return float(100 - 40 * np.mean(np.abs((pr - ob) / len(self.thres))))

    def get_r(self) -> float:
        index = (~np.isnan(self.ob)) & (~np.isnan(self.pr))
        ob = self.ob[index]
        pr = self.pr[index]
        return float(np.corrcoef(ob, pr)[0, 1])

    def get_ts(self) -> np.ndarray:
        ts = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum((self.pr_grade == i) & (self.ob_grade == i))
            nb = np.sum((self.pr_grade == i) & (self.ob_grade != i))
            nc = np.sum((self.pr_grade != i) & (self.ob_grade == i))
            ts[i] = na / (na + nb + nc) if na + nb + nc != 0 else np.nan
        return ts

    def get_ts2(self) -> np.ndarray:
        ts = np.zeros(self.n_grades - 1, dtype=np.float32)
        for i in range(self.n_grades - 1):
            na = np.sum((self.pr_grade > i) & (self.ob_grade > i))
            nb = np.sum((self.pr_grade > i) & (self.ob_grade <= i))
            nc = np.sum((self.pr_grade <= i) & (self.ob_grade > i))
            ts[i] = na / (na + nb + nc) if na + nb + nc != 0 else np.nan
        return ts

    def get_ets(self) -> np.ndarray:
        ets = np.zeros(self.n_grades , dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum((self.pr_grade == i) & (self.ob_grade == i))
            nb = np.sum((self.pr_grade == i) & (self.ob_grade != i))
            nc = np.sum((self.pr_grade != i) & (self.ob_grade == i))
            nd = np.sum((self.pr_grade != i) & (self.ob_grade != i))
            r = (na + nb) / (na + nb + nc + nd) * (na + nc)
            ets[i] = (na - r) / (na + nb + nc - r) if na + nb + nc != 0 else np.nan
        return ets

    def get_ets2(self) -> np.ndarray:
        ets = np.zeros(self.n_grades - 1, dtype=np.float32)
        for i in range(self.n_grades - 1):
            na = np.sum((self.pr_grade > i) & (self.ob_grade > i))
            nb = np.sum((self.pr_grade > i) & (self.ob_grade <= i))
            nc = np.sum((self.pr_grade <= i) & (self.ob_grade > i))
            nd = np.sum((self.pr_grade <= i) & (self.ob_grade <= i))
            r = (na + nb) / (na + nb + nc + nd) * (na + nc)
            ets[i] = (na - r) / (na + nb + nc - r) if na + nb + nc != 0 else np.nan
        return ets

    def get_bias(self) -> np.ndarray:
        bias = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum((self.pr_grade == i) & (self.ob_grade == i))
            nb = np.sum((self.pr_grade == i) & (self.ob_grade != i))
            nc = np.sum((self.pr_grade != i) & (self.ob_grade == i))
            bias[i] = (na + nb) / (na + nc) if na + nc != 0 else np.nan
        return bias

    def get_far(self) -> np.ndarray:
        far = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum((self.pr_grade == i) & (self.ob_grade == i))
            nb = np.sum((self.pr_grade == i) & (self.ob_grade != i))
            far[i] = nb / (na + nb) if na + nb != 0 else np.nan
        return far

    def get_mar(self) -> np.ndarray:
        mar = np.zeros(self.n_grades, dtype=np.float32)
        for i in range(self.n_grades):
            na = np.sum((self.pr_grade == i + 1) & (self.ob_grade == i + 1))
            nc = np.sum((self.pr_grade != i + 1) & (self.ob_grade == i + 1))
            mar[i] = nc / (na + nc) if na + nc != 0 else np.nan
        return mar

    def get_hxjz(self) -> np.ndarray:
        return self.hxjz

    def get_oa(self) -> float:
        return np.sum(np.diag(self.hxjz)) / self.n

    def get_kappa(self) -> float:
        a = np.sum(self.hxjz, axis=1)
        a = a.astype(np.float32)
        b = np.sum(self.hxjz, axis=0)
        b = b.astype(np.float32)
        pe = np.sum(a * b) / self.n / self.n
        return (self.get_oa() - pe) / (1 - pe)


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
    ob = list()
    ec = list()
    pdfm = list()
    mlr = list()
    lgb = list()
    for year in range(2017, 2023):
        ob.append(np.load(rf'D:\Project\wind\97\ob_1_{year}.npy'))
        ec.append(np.load(rf'D:\Project\wind\97\nwp_1_{year}.npy'))
        pdfm.append(np.load(rf'D:\Project\wind\97\pdfm_4_{year}.npy'))
        mlr.append(np.load(rf'D:\Project\wind\97\mlr_4_{year}.npy'))
        lgb.append(np.load(rf'D:\Project\wind\97\lgb_3_{year}.npy'))
    ob = np.vstack(ob)
    ec = np.vstack(ec)
    pdfm = np.vstack(pdfm)
    mlr = np.vstack(mlr)
    lgb = np.vstack(lgb)

    print('ECMWF-IFS')
    acc = Acc(ob, ec)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_oa())
    print(acc.get_kappa())
    print('PDFM')
    acc = Acc(ob, pdfm)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_oa())
    print(acc.get_kappa())
    print('MLR')
    acc = Acc(ob, mlr)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_oa())
    print(acc.get_kappa())
    print('LightGBM')
    acc = Acc(ob, lgb)
    print(acc.get_r())
    print(acc.get_mae())
    print(acc.get_rmse())
    print(acc.get_mre())
    print(acc.get_oa())
    print(acc.get_kappa())

    ts = np.zeros((2, 4, 8), dtype=np.float32) + np.nan
    acc = Acc(ob, ec)
    ts[0, 0, :] = acc.get_ts()[1:]
    ts[1, 0, :] = acc.get_ts2()
    acc = Acc(ob, pdfm)
    ts[0, 1, :] = acc.get_ts()[1:]
    ts[1, 1, :] = acc.get_ts2()
    acc = Acc(ob, mlr)
    ts[0, 2, :] = acc.get_ts()[1:]
    ts[1, 2, :] = acc.get_ts2()
    acc = Acc(ob, lgb)
    ts[0, 3, :] = acc.get_ts()[1:]
    ts[1, 3, :] = acc.get_ts2()
    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(8) - 0.3,
        height=ts[0, 0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(8) - 0.1,
        height=ts[0, 1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(8) + 0.1,
        height=ts[0, 2, :],
        width=0.2,
        color='yellow',
        label='MLR'
    )
    plt.bar(
        x=np.arange(8) + 0.3,
        height=ts[0, 3, :],
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
    plt.legend()
    plt.savefig(r'D:\Project\wind\97\图\ts_grade.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(8) - 0.3,
        height=ts[1, 0, :],
        width=0.2,
        color='blue',
        label='ECMWF-IFS'
    )
    plt.bar(
        x=np.arange(8) - 0.1,
        height=ts[1, 1, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(8) + 0.1,
        height=ts[1, 2, :],
        width=0.2,
        color='yellow',
        label='MLR'
    )
    plt.bar(
        x=np.arange(8) + 0.3,
        height=ts[1, 3, :],
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
    plt.legend()
    plt.savefig(r'D:\Project\wind\97\图\ts_grade+.jpg', bbox_inches='tight', dpi=800)
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
    # rmse_month_vt = np.zeros((5, 12, 24), dtype=np.float32) + np.nan
    # for i in range(12):
    #     for j in range(24):
    #         acc = Acc(ob[month_ind == i + 1, j, :], ec[month_ind == i + 1, j, :])
    #         rmse_month_vt[0, i, j] = acc.get_rmse()
    #         acc = Acc(ob[month_ind == i + 1, j, :], pdfm[month_ind == i + 1, j, :])
    #         rmse_month_vt[1, i, j] = acc.get_rmse()
    #         acc = Acc(ob[month_ind == i + 1, j, :], mlr[month_ind == i + 1, j, :])
    #         rmse_month_vt[2, i, j] = acc.get_rmse()
    #         acc = Acc(ob[month_ind == i + 1, j, :], xgb[month_ind == i + 1, j, :])
    #         rmse_month_vt[3, i, j] = acc.get_rmse()
    #         acc = Acc(ob[month_ind == i + 1, j, :], lgb[month_ind == i + 1, j, :])
    #         rmse_month_vt[4, i, j] = acc.get_rmse()
    # sns.heatmap(rmse_month_vt[0, :, :], cmap='Reds', vmin=0, vmax=1.75, linewidths=0.3)
    # plt.xticks(np.arange(24) + 0.5, [str(x + 1) for x in range(24)])
    # plt.yticks(np.arange(12) + 0.5, [f'{x + 1}月' for x in range(12)], rotation=0)
    # plt.tick_params(axis='x', which='both', bottom=False, top=False)
    # plt.tick_params(axis='y', which='both', left=False, right=False)
    # plt.xlabel('月份')
    # plt.ylabel('预报时效/h')
    # plt.savefig(r'D:\Project\wind\97\图\ec_rmse_month_vt.jpg', bbox_inches='tight', dpi=800)
    # plt.close()
    # sns.heatmap(rmse_month_vt[1, :, :], cmap='Reds', vmin=0, vmax=1.75, linewidths=0.3)
    # plt.xticks(np.arange(24) + 0.5, [str(x + 1) for x in range(24)])
    # plt.yticks(np.arange(12) + 0.5, [f'{x + 1}月' for x in range(12)], rotation=0)
    # plt.tick_params(axis='x', which='both', bottom=False, top=False)
    # plt.tick_params(axis='y', which='both', left=False, right=False)
    # plt.xlabel('月份')
    # plt.ylabel('预报时效/h')
    # plt.savefig(r'D:\Project\wind\97\图\pdfm_rmse_month_vt.jpg', bbox_inches='tight', dpi=800)
    # plt.close()
    # sns.heatmap(rmse_month_vt[2, :, :], cmap='Reds', vmin=0, vmax=1.75, linewidths=0.3)
    # plt.xticks(np.arange(24) + 0.5, [str(x + 1) for x in range(24)])
    # plt.yticks(np.arange(12) + 0.5, [f'{x + 1}月' for x in range(12)], rotation=0)
    # plt.tick_params(axis='x', which='both', bottom=False, top=False)
    # plt.tick_params(axis='y', which='both', left=False, right=False)
    # plt.xlabel('月份')
    # plt.ylabel('预报时效/h')
    # plt.savefig(r'D:\Project\wind\97\图\mlr_rmse_month_vt.jpg', bbox_inches='tight', dpi=800)
    # plt.close()
    # sns.heatmap(rmse_month_vt[3, :, :], cmap='Reds', vmin=0, vmax=1.75, linewidths=0.3)
    # plt.xticks(np.arange(24) + 0.5, [str(x + 1) for x in range(24)])
    # plt.yticks(np.arange(12) + 0.5, [f'{x + 1}月' for x in range(12)], rotation=0)
    # plt.tick_params(axis='x', which='both', bottom=False, top=False)
    # plt.tick_params(axis='y', which='both', left=False, right=False)
    # plt.xlabel('月份')
    # plt.ylabel('预报时效/h')
    # plt.savefig(r'D:\Project\wind\97\图\xgb_rmse_month_vt.jpg', bbox_inches='tight', dpi=800)
    # plt.close()
    # sns.heatmap(rmse_month_vt[4, :, :], cmap='Reds', vmin=0, vmax=1.75, linewidths=0.3)
    # plt.xticks(np.arange(24) + 0.5, [str(x + 1) for x in range(24)])
    # plt.yticks(np.arange(12) + 0.5, [f'{x + 1}月' for x in range(12)], rotation=0)
    # plt.tick_params(axis='x', which='both', bottom=False, top=False)
    # plt.tick_params(axis='y', which='both', left=False, right=False)
    # plt.xlabel('月份')
    # plt.ylabel('预报时效/h')
    # plt.savefig(r'D:\Project\wind\97\图\lgb_rmse_month_vt.jpg', bbox_inches='tight', dpi=800)
    # plt.close()
    # print(np.max(rmse_month_vt))
    rmse_year = np.zeros((4, 6), dtype=np.float32) + np.nan
    rmse_improvement_year = np.zeros((3, 6), dtype=np.float32) + np.nan
    for i in range(6):
        acc = Acc(ob[year_ind == i + 2017, :, :], ec[year_ind == i + 2017, :, :])
        rmse_year[0, i] = acc.get_rmse()
        acc = Acc(ob[year_ind == i + 2017, :, :], pdfm[year_ind == i + 2017, :, :])
        rmse_year[1, i] = acc.get_rmse()
        rmse_improvement_year[0, i] = (rmse_year[0, i] - rmse_year[1, i]) / rmse_year[0, i] * 100
        acc = Acc(ob[year_ind == i + 2017, :, :], mlr[year_ind == i + 2017, :, :])
        rmse_year[2, i] = acc.get_rmse()
        rmse_improvement_year[1, i] = (rmse_year[0, i] - rmse_year[2, i]) / rmse_year[0, i] * 100
        acc = Acc(ob[year_ind == i + 2017, :, :], lgb[year_ind == i + 2017, :, :])
        rmse_year[3, i] = acc.get_rmse()
        rmse_improvement_year[2, i] = (rmse_year[0, i] - rmse_year[3, i]) / rmse_year[0, i] * 100
    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(6) - 0.2,
        height=rmse_improvement_year[0, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(6),
        height=rmse_improvement_year[1, :],
        width=0.2,
        color='yellow',
        label='MLR'
    )
    plt.bar(
        x=np.arange(6) + 0.2,
        height=rmse_improvement_year[2, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 6))
    plt.xticks(range(6), [f'{x + 2017}年' for x in range(6)])
    plt.ylim((0, 45))
    plt.yticks((0, 5, 10, 15, 20, 25, 30, 35, 40, 45))
    plt.xlabel('年份')
    plt.ylabel('RMSE改善率/%')
    plt.legend(loc='upper center', ncols=4)
    plt.savefig(r'D:\Project\wind\97\图\rmse_improvement_year.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    print(np.min(rmse_improvement_year), np.max(rmse_improvement_year))
    rmse_month = np.zeros((4, 12), dtype=np.float32) + np.nan
    rmse_improvement_month = np.zeros((3, 12), dtype=np.float32) + np.nan
    for i in range(12):
        acc = Acc(ob[month_ind == i + 1, :, :], ec[month_ind == i + 1, :, :])
        rmse_month[0, i] = acc.get_rmse()
        acc = Acc(ob[month_ind == i + 1, :, :], pdfm[month_ind == i + 1, :, :])
        rmse_month[1, i] = acc.get_rmse()
        rmse_improvement_month[0, i] = (rmse_month[0, i] - rmse_month[1, i]) / rmse_month[0, i] * 100
        acc = Acc(ob[month_ind == i + 1, :, :], mlr[month_ind == i + 1, :, :])
        rmse_month[2, i] = acc.get_rmse()
        rmse_improvement_month[1, i] = (rmse_month[0, i] - rmse_month[2, i]) / rmse_month[0, i] * 100
        acc = Acc(ob[month_ind == i + 1, :, :], lgb[month_ind == i + 1, :, :])
        rmse_month[3, i] = acc.get_rmse()
        rmse_improvement_month[2, i] = (rmse_month[0, i] - rmse_month[3, i]) / rmse_month[0, i] * 100
    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(12) - 0.2,
        height=rmse_improvement_month[0, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(12),
        height=rmse_improvement_month[1, :],
        width=0.2,
        color='yellow',
        label='MLR'
    )
    plt.bar(
        x=np.arange(12) + 0.2,
        height=rmse_improvement_month[2, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 12))
    plt.xticks(range(12), [f'{x + 1}月' for x in range(12)])
    plt.ylim((0, 45))
    plt.yticks((0, 5, 10, 15, 20, 25, 30, 35, 40, 45))
    plt.xlabel('月份')
    plt.ylabel('RMSE改善率/%')
    plt.legend(loc='upper center', ncols=4)
    plt.savefig(r'D:\Project\wind\97\图\rmse_improvement_month.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    print(np.min(rmse_improvement_month), np.max(rmse_improvement_month))
    rmse_vt = np.zeros((4, 24), dtype=np.float32) + np.nan
    rmse_improvement_vt = np.zeros((3, 24), dtype=np.float32) + np.nan
    for i in range(24):
        acc = Acc(ob[:, i, :], ec[:, i, :])
        rmse_vt[0, i] = acc.get_rmse()
        acc = Acc(ob[:, i, :], pdfm[:, i, :])
        rmse_vt[1, i] = acc.get_rmse()
        rmse_improvement_vt[0, i] = (rmse_vt[0, i] - rmse_vt[1, i]) / rmse_vt[0, i] * 100
        acc = Acc(ob[:, i, :], mlr[:, i, :])
        rmse_vt[2, i] = acc.get_rmse()
        rmse_improvement_vt[1, i] = (rmse_vt[0, i] - rmse_vt[2, i]) / rmse_vt[0, i] * 100
        acc = Acc(ob[:, i, :], lgb[:, i, :])
        rmse_vt[3, i] = acc.get_rmse()
        rmse_improvement_vt[2, i] = (rmse_vt[0, i] - rmse_vt[3, i]) / rmse_vt[0, i] * 100
    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(24) - 0.2,
        height=rmse_improvement_vt[0, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(24),
        height=rmse_improvement_vt[1, :],
        width=0.2,
        color='yellow',
        label='MLR'
    )
    plt.bar(
        x=np.arange(24) + 0.2,
        height=rmse_improvement_vt[2, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 24))
    plt.xticks(range(24), [str(x + 1) for x in range(24)])
    plt.ylim((0, 45))
    plt.yticks((0, 5, 10, 15, 20, 25, 30, 35, 40, 45))
    plt.xlabel('预报时效/h')
    plt.ylabel('RMSE改善率/%')
    plt.legend(loc='upper center', ncols=4)
    plt.savefig(r'D:\Project\wind\97\图\rmse_improvement_vt.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    print(np.min(rmse_improvement_vt), np.max(rmse_improvement_vt))
    rmse_fhour = np.zeros((4, 24), dtype=np.float32) + np.nan
    rmse_improvement_fhour = np.zeros((3, 24), dtype=np.float32) + np.nan
    for i in range(24):
        acc = Acc(ob[fhour_ind == i, :], ec[fhour_ind == i, :])
        rmse_fhour[0, i] = acc.get_rmse()
        acc = Acc(ob[fhour_ind == i, :], pdfm[fhour_ind == i, :])
        rmse_fhour[1, i] = acc.get_rmse()
        rmse_improvement_fhour[0, i] = (rmse_fhour[0, i] - rmse_fhour[1, i]) / rmse_fhour[0, i] * 100
        acc = Acc(ob[fhour_ind == i, :], mlr[fhour_ind == i, :])
        rmse_fhour[2, i] = acc.get_rmse()
        rmse_improvement_fhour[1, i] = (rmse_fhour[0, i] - rmse_fhour[2, i]) / rmse_fhour[0, i] * 100
        acc = Acc(ob[fhour_ind == i, :], lgb[fhour_ind == i, :])
        rmse_fhour[3, i] = acc.get_rmse()
        rmse_improvement_fhour[2, i] = (rmse_fhour[0, i] - rmse_fhour[3, i]) / rmse_fhour[0, i] * 100
    plt.figure(figsize=(15, 5))
    plt.bar(
        x=np.arange(24) - 0.2,
        height=rmse_improvement_fhour[0, :],
        width=0.2,
        color='green',
        label='PDFM'
    )
    plt.bar(
        x=np.arange(24),
        height=rmse_improvement_fhour[1, :],
        width=0.2,
        color='yellow',
        label='MLR'
    )
    plt.bar(
        x=np.arange(24) + 0.2,
        height=rmse_improvement_fhour[2, :],
        width=0.2,
        color='red',
        label='LightGBM'
    )
    plt.xlim((-1, 24))
    plt.xticks(range(24), [f'{x}:00' for x in range(24)])
    plt.ylim((0, 45))
    plt.yticks((0, 5, 10, 15, 20, 25, 30, 35, 40, 45))
    plt.xlabel('预报时间/UTC')
    plt.ylabel('RMSE改善率/%')
    plt.legend(loc='upper center', ncols=4)
    plt.savefig(r'D:\Project\wind\97\图\rmse_improvement_fhour.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    print(np.min(rmse_improvement_fhour), np.max(rmse_improvement_fhour))

    station_information = pd.read_csv(r'D:\Project\wind\国家气象观测站.csv', encoding='gb2312', low_memory=False)
    station_information.sort_values(by=['台站号'], inplace=True)
    station_information.reset_index(drop=True, inplace=True)
    sta = station_information.loc[:, ('台站号', '经度', '纬度')]
    sta = meb.sta_data(sta, columns=['id', 'lon', 'lat'])
    sta.loc[:, 'level'] = 0
    sta.loc[:, 'time'] = '2024-01-01 08:00:00'
    sta.loc[:, 'dtime'] = 0
    rmse_sta = np.zeros((4, 97), dtype=np.float32) + np.nan
    rmse_improvement_sta = np.zeros((3, 97), dtype=np.float32) + np.nan
    for i in range(97):
        acc = Acc(ob[:, :, i], ec[:, :, i])
        rmse_sta[0, i] = acc.get_rmse()
        acc = Acc(ob[:, :, i], pdfm[:, :, i])
        rmse_sta[1, i] = acc.get_rmse()
        rmse_improvement_sta[0, i] = (rmse_sta[0, i] - rmse_sta[1, i]) / rmse_sta[0, i] * 100
        acc = Acc(ob[:, :, i], mlr[:, :, i])
        rmse_sta[2, i] = acc.get_rmse()
        rmse_improvement_sta[1, i] = (rmse_sta[0, i] - rmse_sta[2, i]) / rmse_sta[0, i] * 100
        acc = Acc(ob[:, :, i], lgb[:, :, i])
        rmse_sta[3, i] = acc.get_rmse()
        rmse_improvement_sta[2, i] = (rmse_sta[0, i] - rmse_sta[3, i]) / rmse_sta[0, i] * 100
    cmap, clevs = meb.def_cmap_clevs(
        meb.def_cmap_clevs(meb.cmaps.mae, vmin=-20, vmax=80)[0],
        clevs=[-20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80]
    )
    sta.loc[:, 'data0'] = rmse_improvement_sta[0, :]
    meb.tool.plot_tools.scatter_sta(
        sta0=sta,
        map_extend=[108.65, 114.4, 24.5, 30.25],
        clevs=clevs,
        cmap=cmap,
        title=[''],
        save_path=r'D:\Project\wind\97\图\pdfm_rmse_improvement_sta.jpg',
        dpi=800
    )
    sta.loc[:, 'data0'] = rmse_improvement_sta[1, :]
    meb.tool.plot_tools.scatter_sta(
        sta0=sta,
        map_extend=[108.65, 114.4, 24.5, 30.25],
        clevs=clevs,
        cmap=cmap,
        title=[''],
        save_path=r'D:\Project\wind\97\图\mlr_rmse_improvement_sta.jpg',
        dpi=800
    )
    sta.loc[:, 'data0'] = rmse_improvement_sta[2, :]
    meb.tool.plot_tools.scatter_sta(
        sta0=sta,
        map_extend=[108.65, 114.4, 24.5, 30.25],
        clevs=clevs,
        cmap=cmap,
        title=[''],
        save_path=r'D:\Project\wind\97\图\lgb_rmse_improvement_sta.jpg',
        dpi=800
    )
    print(np.min(rmse_improvement_sta, axis=1))
    print(np.mean(rmse_improvement_sta, axis=1))
    print(np.median(rmse_improvement_sta, axis=1))
    print(np.max(rmse_improvement_sta, axis=1))

    plt.figure(figsize=(5, 5))
    plt.scatter(
        x=station_information.loc[:, '经度'],
        y=rmse_improvement_sta[0, :],
        s=0.2,
        c='green',
        label='PDFM'
    )
    plt.scatter(
        x=station_information.loc[:, '经度'],
        y=rmse_improvement_sta[1, :],
        s=0.2,
        c='yellow',
        label='MLR'
    )
    plt.scatter(
        x=station_information.loc[:, '经度'],
        y=rmse_improvement_sta[2, :],
        s=0.2,
        c='red',
        label='LightGBM'
    )
    plt.xlim((108.65, 114.4))
    plt.xticks([109, 110, 111, 112, 113, 114], ['109°', '110°', '111°', '112°', '113°', '114°E'])
    plt.ylim((-20, 80))
    plt.yticks((-20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80))
    plt.xlabel('经度')
    plt.ylabel('RMSE改善率/m')
    plt.legend()
    plt.savefig(r'D:\Project\wind\97\图\rmse_improvement-lon.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    print('经度')
    print(np.corrcoef(station_information.loc[:, '经度'], rmse_improvement_sta[0, :])[0, 1])
    print(np.corrcoef(station_information.loc[:, '经度'], rmse_improvement_sta[1, :])[0, 1])
    print(np.corrcoef(station_information.loc[:, '经度'], rmse_improvement_sta[2, :])[0, 1])
    plt.figure(figsize=(5, 5))
    plt.scatter(
        x=station_information.loc[:, '纬度'],
        y=rmse_improvement_sta[0, :],
        s=0.2,
        c='green',
        label='PDFM'
    )
    plt.scatter(
        x=station_information.loc[:, '纬度'],
        y=rmse_improvement_sta[1, :],
        s=0.2,
        c='yellow',
        label='MLR'
    )
    plt.scatter(
        x=station_information.loc[:, '纬度'],
        y=rmse_improvement_sta[2, :],
        s=0.2,
        c='red',
        label='LightGBM'
    )
    plt.xlim((24.5, 30.25))
    plt.xticks([25, 26, 27, 28, 29, 30], ['25°', '26°', '27°', '28°', '29°', '30°N'])
    plt.ylim((-20, 80))
    plt.yticks((-20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80))
    plt.xlabel('纬度')
    plt.ylabel('RMSE改善率/m')
    plt.legend()
    plt.savefig(r'D:\Project\wind\97\图\rmse_improvement-lat.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    print('纬度')
    print(np.corrcoef(station_information.loc[:, '纬度'], rmse_improvement_sta[0, :])[0, 1])
    print(np.corrcoef(station_information.loc[:, '纬度'], rmse_improvement_sta[1, :])[0, 1])
    print(np.corrcoef(station_information.loc[:, '纬度'], rmse_improvement_sta[2, :])[0, 1])
    plt.figure(figsize=(5, 5))
    plt.scatter(
        x=station_information.loc[:, '海拔'],
        y=rmse_improvement_sta[0, :],
        s=0.2,
        c='green',
        label='PDFM'
    )
    plt.scatter(
        x=station_information.loc[:, '海拔'],
        y=rmse_improvement_sta[1, :],
        s=0.2,
        c='yellow',
        label='MLR'
    )
    plt.scatter(
        x=station_information.loc[:, '海拔'],
        y=rmse_improvement_sta[2, :],
        s=0.2,
        c='red',
        label='LightGBM'
    )
    plt.xlim((0, 1500))
    plt.xticks([0, 300, 600, 900, 1200, 1500], ['0', '300', '600', '900', '1200', '1500'])
    plt.ylim((-20, 80))
    plt.yticks((-20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80))
    plt.xlabel('高程/m')
    plt.ylabel('RMSE改善率/m')
    plt.legend()
    plt.savefig(r'D:\Project\wind\97\图\rmse_improvement-alti.jpg', bbox_inches='tight', dpi=800)
    plt.close()
    print('高程')
    print(np.corrcoef(station_information.loc[:, '海拔'], rmse_improvement_sta[0, :])[0, 1])
    print(np.corrcoef(station_information.loc[:, '海拔'], rmse_improvement_sta[1, :])[0, 1])
    print(np.corrcoef(station_information.loc[:, '海拔'], rmse_improvement_sta[2, :])[0, 1])

    plt.figure(figsize=(5, 5))
    sns.violinplot(
        data={
            'PDFM': rmse_improvement_sta[0, :],
            'MLR': rmse_improvement_sta[1, :],
            'LightGBM': rmse_improvement_sta[2, :]
        },
        palette=['green', 'yellow', 'red']
    )
    plt.xlabel('模型')
    plt.ylabel('RMSE改善率/m')
    plt.savefig(r'D:\Project\wind\97\图\rmse_improvement_boxplot.jpg', bbox_inches='tight', dpi=800)
    plt.close()


if __name__ == '__main__':
    print('The program "access97.py" is beginning.')
    start = arrow.now()

    main()

    end = arrow.now()
    running_time = (end - start).total_seconds()

    print('The program "access97.py" runs out in {:s}.'.format(format_time(running_time)))
