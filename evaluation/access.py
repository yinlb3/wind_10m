#!user/bin.python3

"""
Founded in 2023-07-19
Modified in 2023-08-06
@author: yinlb
"""
import datetime
import os
import sys

import arrow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from meteva import base as meb
from meteva import method as mem
from scipy import stats


def draw_scatter(x: np.ndarray, y: np.ndarray, save_path: str) -> None:
    # Calculate the point density
    xy = np.vstack([x, y])
    z = stats.gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    print(np.sum(idx))

    fig, ax = plt.subplots()
    plt.scatter(x, y, c=z, s=20, cmap='Spectral')
    plt.colorbar()
    plt.savefig(save_path, dpi=300)
    plt.cla()


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


def main(input_path: str, output_path: str, station_information_path: str):
    station_information = pd.read_csv(station_information_path, encoding='gb2312', low_memory=False)
    station = meb.sta_data(station_information.loc[:, ('台站号', '经度', '纬度', '海拔')],
                           columns=('id', 'lon', 'lat', 'e'))
    station.sort_values(by=['id'], inplace=True)
    station.reset_index(drop=True, inplace=True)

    ob_label = list()
    lgb_label = list()
    nwp_label = list()
    for year in range(2017, 2023):
        temp_data = np.load(os.path.join(input_path, 'ob_{:d}.npy'.format(year)))
        if len(temp_data.shape) == 4:
            temp_data = temp_data[:, :, :, 1]
        ob_label.append(temp_data)
        temp_data = np.load(os.path.join(input_path, 'lgb_{:d}.npy'.format(year)))
        if len(temp_data.shape) == 4:
            temp_data = temp_data[:, :, :, 1]
        lgb_label.append(temp_data)
        temp_data = np.load(os.path.join(input_path, 'nwp_{:d}.npy'.format(year)))
        if len(temp_data.shape) == 4:
            temp_data = temp_data[:, :, :, 1]
        nwp_label.append(temp_data)
    ob_label = np.vstack(ob_label)
    lgb_label = np.vstack(lgb_label)
    nwp_label = np.vstack(nwp_label)
    data_time = np.zeros_like(ob_label, dtype='datetime64[ns]')
    for i in range(4382):
        data_time[i, :, :] = arrow.get('2017').shift(hours=12 * i).datetime
    data_dtime = np.zeros_like(ob_label, dtype=np.int_)
    for i in range(25):
        data_dtime[:, i, :] = 12 + i
    data_id = np.zeros_like(ob_label, dtype=np.int_)
    data_e = np.zeros_like(ob_label, dtype=np.float_)
    for i in range(97):
        data_id[:, :, i] = station.loc[i, 'id']
        data_e[:, :, i] = station.loc[i, 'e']
    sta = meb.sta_data(df=pd.DataFrame(data={'time': np.reshape(data_time, -1), 'dtime': np.reshape(data_dtime, -1),
                                             'id': np.reshape(data_id, -1), 'ob': np.reshape(ob_label, -1),
                                             'NWP': np.reshape(nwp_label, -1), 'LightGBM': np.reshape(lgb_label, -1)}),
                       columns=['time', 'dtime', 'id', 'ob', 'NWP', 'LightGBM'])
    for i in range(97):
        sta.loc[sta.loc[:, 'id'] == station.loc[i, 'id'], 'lon'] = station.loc[i, 'lon']
        sta.loc[sta.loc[:, 'id'] == station.loc[i, 'id'], 'lat'] = station.loc[i, 'lat']
    meb.set_stadata_coords(sta, level=0)
    index = ~np.isnan(sta.loc[:, 'ob']) & ~np.isnan(sta.loc[:, 'NWP']) & ~np.isnan(sta.loc[:, 'LightGBM'])
    sta = sta.loc[index]
    sta.reset_index(drop=True, inplace=True)
    print(len(sta))
    print(sta.max())

    # draw_scatter(np.array(sta.loc[:, 'ob']), np.array(sta.loc[:, 'NWP']),
    #              save_path=os.path.join(output_path, 'EC散点密度图.png'))
    # draw_scatter(np.array(sta.loc[:, 'ob']), np.array(sta.loc[:, 'LightGBM']),
    #              save_path=os.path.join(output_path, 'LightGBM散点密度图.png'))
    # mem.scatter_regress(np.array(sta.loc[:, 'ob']).T, np.array(sta.loc[:, ('NWP', 'LightGBM')]).T,
    #                     save_path=os.path.join(output_path, '散点回归图.png'), dpi=300)
    # mem.pdf_plot(np.array(sta.loc[:, 'ob']).T, np.array(sta.loc[:, ('NWP', 'LightGBM')]).T,
    #              save_path=os.path.join(output_path, '频率关系图.png'), dpi=300)
    # mem.box_plot_continue(np.array(sta.loc[:, 'ob']).T, np.array(sta.loc[:, ('NWP', 'LightGBM')]).T,
    #                       save_path=os.path.join(output_path, '频率湘线图.png'), dpi=300)

    # 分时次
    dic = {'时次': list(), 'R_EC': list(), 'ME_EC': list(), 'MAE_EC': list(), 'RMSE_EC': list(), 'MRE_EC': list(),
           'R_LightGBM': list(), 'ME_LightGBM': list(), 'MAE_LightGBM': list(), 'RMSE_LightGBM': list(),
           'MRE_LightGBM': list()}
    ob = np.array(sta.loc[:, 'ob']).T
    fo = np.array(sta.loc[:, ('NWP', 'LightGBM')]).T
    r = mem.corr(ob, fo)
    me = mem.me(ob, fo)
    mae = mem.mae(ob, fo)
    rmse = mem.rmse(ob, fo)
    mre = mem.mre(ob, fo)
    dic['时次'].append('All')
    dic['R_EC'].append(r[0])
    dic['ME_EC'].append(me[0])
    dic['MAE_EC'].append(mae[0])
    dic['RMSE_EC'].append(rmse[0])
    dic['MRE_EC'].append(mre[0])
    dic['R_LightGBM'].append(r[1])
    dic['ME_LightGBM'].append(me[1])
    dic['MAE_LightGBM'].append(mae[1])
    dic['RMSE_LightGBM'].append(rmse[1])
    dic['MRE_LightGBM'].append(mre[1])
    for i in range(25):
        sta0 = meb.sele_by_para(sta, dtime=12 + i)
        ob = np.array(sta0.loc[:, 'ob']).T
        fo = np.array(sta0.loc[:, ('NWP', 'LightGBM')]).T
        r = mem.corr(ob, fo)
        me = mem.me(ob, fo)
        mae = mem.mae(ob, fo)
        rmse = mem.rmse(ob, fo)
        mre = mem.mre(ob, fo)
        dic['时次'].append(str(12 + i))
        dic['R_EC'].append(r[0])
        dic['ME_EC'].append(me[0])
        dic['MAE_EC'].append(mae[0])
        dic['RMSE_EC'].append(rmse[0])
        dic['MRE_EC'].append(mre[0])
        dic['R_LightGBM'].append(r[1])
        dic['ME_LightGBM'].append(me[1])
        dic['MAE_LightGBM'].append(mae[1])
        dic['RMSE_LightGBM'].append(rmse[1])
        dic['MRE_LightGBM'].append(mre[1])
    df = pd.DataFrame(data=dic, columns=['时次', 'R_EC', 'ME_EC', 'MAE_EC', 'RMSE_EC', 'MRE_EC', 'R_LightGBM',
                                         'ME_LightGBM', 'MAE_LightGBM', 'RMSE_LightGBM', 'MRE_LightGBM'])
    plt.bar(np.arange(10.8, 36.7, 1), df.loc[:, 'R_EC'], width=0.4, color='blue', label='EC')
    plt.bar(np.arange(11.2, 37.1, 1), df.loc[:, 'R_LightGBM'], width=0.4, color='red', label='LightGBM')
    plt.xticks(range(11, 37), df.loc[:, '时次'])
    plt.title('R')
    plt.legend()
    plt.savefig(os.path.join(output_path, '分时次_r.png'), dpi=300)
    plt.cla()
    plt.bar(np.arange(10.8, 36.7, 1), df.loc[:, 'ME_EC'], width=0.4, color='blue', label='EC')
    plt.bar(np.arange(11.2, 37.1, 1), df.loc[:, 'ME_LightGBM'], width=0.4, color='red', label='LightGBM')
    plt.xticks(range(11, 37), df.loc[:, '时次'])
    plt.title('ME')
    plt.legend()
    plt.savefig(os.path.join(output_path, '分时次_me.png'), dpi=300)
    plt.cla()
    plt.bar(np.arange(10.8, 36.7, 1), df.loc[:, 'MAE_EC'], width=0.4, color='blue', label='EC')
    plt.bar(np.arange(11.2, 37.1, 1), df.loc[:, 'MAE_LightGBM'], width=0.4, color='red', label='LightGBM')
    plt.xticks(range(11, 37), df.loc[:, '时次'])
    plt.title('MAE')
    plt.legend()
    plt.savefig(os.path.join(output_path, '分时次_mae.png'), dpi=300)
    plt.cla()
    plt.bar(np.arange(10.8, 36.7, 1), df.loc[:, 'RMSE_EC'], width=0.4, color='blue', label='EC')
    plt.bar(np.arange(11.2, 37.1, 1), df.loc[:, 'RMSE_LightGBM'], width=0.4, color='red', label='LightGBM')
    plt.xticks(range(11, 37), df.loc[:, '时次'])
    plt.title('RMSE')
    plt.legend()
    plt.savefig(os.path.join(output_path, '分时次_rmse.png'), dpi=300)
    plt.cla()
    plt.bar(np.arange(10.8, 36.7, 1), df.loc[:, 'MRE_EC'], width=0.4, color='blue', label='EC')
    plt.bar(np.arange(11.2, 37.1, 1), df.loc[:, 'MRE_LightGBM'], width=0.4, color='red', label='LightGBM')
    plt.xticks(range(11, 37), df.loc[:, '时次'])
    plt.title('MRE')
    plt.legend()
    plt.savefig(os.path.join(output_path, '分时次_mre.png'), dpi=300)
    plt.cla()
    df.to_csv(os.path.join(output_path, '分时次.csv'), encoding='gb2312', index=False)

    # 分月份
    dic = {'月份': list(), 'R_EC': list(), 'ME_EC': list(), 'MAE_EC': list(), 'RMSE_EC': list(), 'MRE_EC': list(),
           'R_LightGBM': list(), 'ME_LightGBM': list(), 'MAE_LightGBM': list(), 'RMSE_LightGBM': list(),
           'MRE_LightGBM': list()}
    ob = np.array(sta.loc[:, 'ob']).T
    fo = np.array(sta.loc[:, ('NWP', 'LightGBM')]).T
    r = mem.corr(ob, fo)
    me = mem.me(ob, fo)
    mae = mem.mae(ob, fo)
    rmse = mem.rmse(ob, fo)
    mre = mem.mre(ob, fo)
    dic['月份'].append('All')
    dic['R_EC'].append(r[0])
    dic['ME_EC'].append(me[0])
    dic['MAE_EC'].append(mae[0])
    dic['RMSE_EC'].append(rmse[0])
    dic['MRE_EC'].append(mre[0])
    dic['R_LightGBM'].append(r[1])
    dic['ME_LightGBM'].append(me[1])
    dic['MAE_LightGBM'].append(mae[1])
    dic['RMSE_LightGBM'].append(rmse[1])
    dic['MRE_LightGBM'].append(mre[1])
    for i in range(12):
        sta0 = meb.sele_by_para(sta, month=i + 1)
        ob = np.array(sta0.loc[:, 'ob']).T
        fo = np.array(sta0.loc[:, ('NWP', 'LightGBM')]).T
        r = mem.corr(ob, fo)
        me = mem.me(ob, fo)
        mae = mem.mae(ob, fo)
        rmse = mem.rmse(ob, fo)
        mre = mem.mre(ob, fo)
        dic['月份'].append(str(i + 1))
        dic['R_EC'].append(r[0])
        dic['ME_EC'].append(me[0])
        dic['MAE_EC'].append(mae[0])
        dic['RMSE_EC'].append(rmse[0])
        dic['MRE_EC'].append(mre[0])
        dic['R_LightGBM'].append(r[1])
        dic['ME_LightGBM'].append(me[1])
        dic['MAE_LightGBM'].append(mae[1])
        dic['RMSE_LightGBM'].append(rmse[1])
        dic['MRE_LightGBM'].append(mre[1])
    df = pd.DataFrame(data=dic, columns=['月份', 'R_EC', 'ME_EC', 'MAE_EC', 'RMSE_EC', 'MRE_EC', 'R_LightGBM',
                                         'ME_LightGBM', 'MAE_LightGBM', 'RMSE_LightGBM', 'MRE_LightGBM'])
    plt.bar(np.arange(-0.2, 12.8, 1), df.loc[:, 'R_EC'], width=0.4, color='blue', label='EC')
    plt.bar(np.arange(0.2, 13.2, 1), df.loc[:, 'R_LightGBM'], width=0.4, color='red', label='LightGBM')
    plt.xticks(range(13), df.loc[:, '月份'])
    plt.title('R')
    plt.legend()
    plt.savefig(os.path.join(output_path, '分月份_r.png'), dpi=300)
    plt.cla()
    plt.bar(np.arange(-0.2, 12.8, 1), df.loc[:, 'ME_EC'], width=0.4, color='blue', label='EC')
    plt.bar(np.arange(0.2, 13.2, 1), df.loc[:, 'ME_LightGBM'], width=0.4, color='red', label='LightGBM')
    plt.xticks(range(13), df.loc[:, '月份'])
    plt.title('ME')
    plt.legend()
    plt.savefig(os.path.join(output_path, '分月份_me.png'), dpi=300)
    plt.cla()
    plt.bar(np.arange(-0.2, 12.8, 1), df.loc[:, 'MAE_EC'], width=0.4, color='blue', label='EC')
    plt.bar(np.arange(0.2, 13.2, 1), df.loc[:, 'MAE_LightGBM'], width=0.4, color='red', label='LightGBM')
    plt.xticks(range(13), df.loc[:, '月份'])
    plt.title('MAE')
    plt.legend()
    plt.savefig(os.path.join(output_path, '分月份_mae.png'), dpi=300)
    plt.cla()
    plt.bar(np.arange(-0.2, 12.8, 1), df.loc[:, 'RMSE_EC'], width=0.4, color='blue', label='EC')
    plt.bar(np.arange(0.2, 13.2, 1), df.loc[:, 'RMSE_LightGBM'], width=0.4, color='red', label='LightGBM')
    plt.xticks(range(13), df.loc[:, '月份'])
    plt.title('RMSE')
    plt.legend()
    plt.savefig(os.path.join(output_path, '分月份_rmse.png'), dpi=300)
    plt.cla()
    plt.bar(np.arange(-0.2, 12.8, 1), df.loc[:, 'MRE_EC'], width=0.4, color='blue', label='EC')
    plt.bar(np.arange(0.2, 13.2, 1), df.loc[:, 'MRE_LightGBM'], width=0.4, color='red', label='LightGBM')
    plt.xticks(range(13), df.loc[:, '月份'])
    plt.title('MRE')
    plt.legend()
    plt.savefig(os.path.join(output_path, '分月份_mre.png'), dpi=300)
    plt.cla()
    df.to_csv(os.path.join(output_path, '分月份.csv'), encoding='gb2312', index=False)

    # 分量级
    dic = {'量级': list(), 'R_EC': list(), 'ME_EC': list(), 'MAE_EC': list(), 'RMSE_EC': list(), 'MRE_EC': list(),
           'R_LightGBM': list(), 'ME_LightGBM': list(), 'MAE_LightGBM': list(), 'RMSE_LightGBM': list(),
           'MRE_LightGBM': list()}
    ob = np.array(sta.loc[:, 'ob']).T
    fo = np.array(sta.loc[:, ('NWP', 'LightGBM')]).T
    r = mem.corr(ob, fo)
    me = mem.me(ob, fo)
    mae = mem.mae(ob, fo)
    rmse = mem.rmse(ob, fo)
    mre = mem.mre(ob, fo)
    dic['量级'].append('All')
    dic['R_EC'].append(r[0])
    dic['ME_EC'].append(me[0])
    dic['MAE_EC'].append(mae[0])
    dic['RMSE_EC'].append(rmse[0])
    dic['MRE_EC'].append(mre[0])
    dic['R_LightGBM'].append(r[1])
    dic['ME_LightGBM'].append(me[1])
    dic['MAE_LightGBM'].append(mae[1])
    dic['RMSE_LightGBM'].append(rmse[1])
    dic['MRE_LightGBM'].append(mre[1])
    index = (sta.loc[:, 'ob'] >= 0.3) & (sta.loc[:, 'ob'] < 1.6)
    sta0 = sta.loc[index]
    ob = np.array(sta0.loc[:, 'ob']).T
    fo = np.array(sta0.loc[:, ('NWP', 'LightGBM')]).T
    r = mem.corr(ob, fo)
    me = mem.me(ob, fo)
    mae = mem.mae(ob, fo)
    rmse = mem.rmse(ob, fo)
    mre = mem.mre(ob, fo)
    dic['量级'].append('1级')
    dic['R_EC'].append(r[0])
    dic['ME_EC'].append(me[0])
    dic['MAE_EC'].append(mae[0])
    dic['RMSE_EC'].append(rmse[0])
    dic['MRE_EC'].append(mre[0])
    dic['R_LightGBM'].append(r[1])
    dic['ME_LightGBM'].append(me[1])
    dic['MAE_LightGBM'].append(mae[1])
    dic['RMSE_LightGBM'].append(rmse[1])
    dic['MRE_LightGBM'].append(mre[1])
    index = (sta.loc[:, 'ob'] >= 1.6) & (sta.loc[:, 'ob'] < 3.4)
    sta0 = sta.loc[index]
    ob = np.array(sta0.loc[:, 'ob']).T
    fo = np.array(sta0.loc[:, ('NWP', 'LightGBM')]).T
    r = mem.corr(ob, fo)
    me = mem.me(ob, fo)
    mae = mem.mae(ob, fo)
    rmse = mem.rmse(ob, fo)
    mre = mem.mre(ob, fo)
    dic['量级'].append('2级')
    dic['R_EC'].append(r[0])
    dic['ME_EC'].append(me[0])
    dic['MAE_EC'].append(mae[0])
    dic['RMSE_EC'].append(rmse[0])
    dic['MRE_EC'].append(mre[0])
    dic['R_LightGBM'].append(r[1])
    dic['ME_LightGBM'].append(me[1])
    dic['MAE_LightGBM'].append(mae[1])
    dic['RMSE_LightGBM'].append(rmse[1])
    dic['MRE_LightGBM'].append(mre[1])
    index = (sta.loc[:, 'ob'] >= 3.4) & (sta.loc[:, 'ob'] < 5.5)
    sta0 = sta.loc[index]
    ob = np.array(sta0.loc[:, 'ob']).T
    fo = np.array(sta0.loc[:, ('NWP', 'LightGBM')]).T
    r = mem.corr(ob, fo)
    me = mem.me(ob, fo)
    mae = mem.mae(ob, fo)
    rmse = mem.rmse(ob, fo)
    mre = mem.mre(ob, fo)
    dic['量级'].append('3级')
    dic['R_EC'].append(r[0])
    dic['ME_EC'].append(me[0])
    dic['MAE_EC'].append(mae[0])
    dic['RMSE_EC'].append(rmse[0])
    dic['MRE_EC'].append(mre[0])
    dic['R_LightGBM'].append(r[1])
    dic['ME_LightGBM'].append(me[1])
    dic['MAE_LightGBM'].append(mae[1])
    dic['RMSE_LightGBM'].append(rmse[1])
    dic['MRE_LightGBM'].append(mre[1])
    index = (sta.loc[:, 'ob'] >= 5.5) & (sta.loc[:, 'ob'] < 8.0)
    sta0 = sta.loc[index]
    ob = np.array(sta0.loc[:, 'ob']).T
    fo = np.array(sta0.loc[:, ('NWP', 'LightGBM')]).T
    r = mem.corr(ob, fo)
    me = mem.me(ob, fo)
    mae = mem.mae(ob, fo)
    rmse = mem.rmse(ob, fo)
    mre = mem.mre(ob, fo)
    dic['量级'].append('4级')
    dic['R_EC'].append(r[0])
    dic['ME_EC'].append(me[0])
    dic['MAE_EC'].append(mae[0])
    dic['RMSE_EC'].append(rmse[0])
    dic['MRE_EC'].append(mre[0])
    dic['R_LightGBM'].append(r[1])
    dic['ME_LightGBM'].append(me[1])
    dic['MAE_LightGBM'].append(mae[1])
    dic['RMSE_LightGBM'].append(rmse[1])
    dic['MRE_LightGBM'].append(mre[1])
    index = sta.loc[:, 'ob'] >= 8.0
    sta0 = sta.loc[index]
    ob = np.array(sta0.loc[:, 'ob']).T
    fo = np.array(sta0.loc[:, ('NWP', 'LightGBM')]).T
    r = mem.corr(ob, fo)
    me = mem.me(ob, fo)
    mae = mem.mae(ob, fo)
    rmse = mem.rmse(ob, fo)
    mre = mem.mre(ob, fo)
    dic['量级'].append('≥5级')
    dic['R_EC'].append(r[0])
    dic['ME_EC'].append(me[0])
    dic['MAE_EC'].append(mae[0])
    dic['RMSE_EC'].append(rmse[0])
    dic['MRE_EC'].append(mre[0])
    dic['R_LightGBM'].append(r[1])
    dic['ME_LightGBM'].append(me[1])
    dic['MAE_LightGBM'].append(mae[1])
    dic['RMSE_LightGBM'].append(rmse[1])
    dic['MRE_LightGBM'].append(mre[1])
    df = pd.DataFrame(data=dic, columns=['量级', 'R_EC', 'ME_EC', 'MAE_EC', 'RMSE_EC', 'MRE_EC', 'R_LightGBM',
                                         'ME_LightGBM', 'MAE_LightGBM', 'RMSE_LightGBM', 'MRE_LightGBM'])
    plt.bar(np.arange(-0.2, 5.8, 1), df.loc[:, 'R_EC'], width=0.4, color='blue', label='EC')
    plt.bar(np.arange(0.2, 6.2, 1), df.loc[:, 'R_LightGBM'], width=0.4, color='red', label='LightGBM')
    plt.xticks(range(6), df.loc[:, '量级'])
    plt.title('R')
    plt.legend()
    plt.savefig(os.path.join(output_path, '分量级_r.png'), dpi=300)
    plt.cla()
    plt.bar(np.arange(-0.2, 5.8, 1), df.loc[:, 'ME_EC'], width=0.4, color='blue', label='EC')
    plt.bar(np.arange(0.2, 6.2, 1), df.loc[:, 'ME_LightGBM'], width=0.4, color='red', label='LightGBM')
    plt.xticks(range(6), df.loc[:, '量级'])
    plt.title('ME')
    plt.legend()
    plt.savefig(os.path.join(output_path, '分量级_me.png'), dpi=300)
    plt.cla()
    plt.bar(np.arange(-0.2, 5.8, 1), df.loc[:, 'MAE_EC'], width=0.4, color='blue', label='EC')
    plt.bar(np.arange(0.2, 6.2, 1), df.loc[:, 'MAE_LightGBM'], width=0.4, color='red', label='LightGBM')
    plt.xticks(range(6), df.loc[:, '量级'])
    plt.title('MAE')
    plt.legend()
    plt.savefig(os.path.join(output_path, '分量级_mae.png'), dpi=300)
    plt.cla()
    plt.bar(np.arange(-0.2, 5.8, 1), df.loc[:, 'RMSE_EC'], width=0.4, color='blue', label='EC')
    plt.bar(np.arange(0.2, 6.2, 1), df.loc[:, 'RMSE_LightGBM'], width=0.4, color='red', label='LightGBM')
    plt.xticks(range(6), df.loc[:, '量级'])
    plt.title('RMSE')
    plt.legend()
    plt.savefig(os.path.join(output_path, '分量级_rmse.png'), dpi=300)
    plt.cla()
    plt.bar(np.arange(-0.2, 5.8, 1), df.loc[:, 'MRE_EC'], width=0.4, color='blue', label='EC')
    plt.bar(np.arange(0.2, 6.2, 1), df.loc[:, 'MRE_LightGBM'], width=0.4, color='red', label='LightGBM')
    plt.xticks(range(6), df.loc[:, '量级'])
    plt.title('MRE')
    plt.legend()
    plt.savefig(os.path.join(output_path, '分量级_mre.png'), dpi=300)
    plt.cla()
    df.to_csv(os.path.join(output_path, '分量级.csv'), encoding='gb2312', index=False)

    # 分站点
    df = station.copy()
    df.loc[:, 'R_EC'] = 0.0
    df.loc[:, 'ME_EC'] = 0.0
    df.loc[:, 'MAE_EC'] = 0.0
    df.loc[:, 'RMSE_EC'] = 0.0
    df.loc[:, 'MRE_EC'] = 0.0
    df.loc[:, 'R_LightGBM'] = 0.0
    df.loc[:, 'ME_LightGBM'] = 0.0
    df.loc[:, 'MAE_LightGBM'] = 0.0
    df.loc[:, 'RMSE_LightGBM'] = 0.0
    df.loc[:, 'MRE_LightGBM'] = 0.0
    for i in range(97):
        sta0 = meb.sele_by_para(sta, id=station.loc[i, 'id'])
        ob = np.array(sta0.loc[:, 'ob']).T
        fo = np.array(sta0.loc[:, ('NWP', 'LightGBM')]).T
        r = mem.corr(ob, fo)
        me = mem.me(ob, fo)
        mae = mem.mae(ob, fo)
        rmse = mem.rmse(ob, fo)
        mre = mem.mre(ob, fo)
        df.loc[i, 'R_EC'] = r[0]
        df.loc[i, 'ME_EC'] = me[0]
        df.loc[i, 'MAE_EC'] = mae[0]
        df.loc[i, 'RMSE_EC'] = rmse[0]
        df.loc[i, 'MRE_EC'] = mre[0]
        df.loc[i, 'R_LightGBM'] = r[1]
        df.loc[i, 'ME_LightGBM'] = me[1]
        df.loc[i, 'MAE_LightGBM'] = mae[1]
        df.loc[i, 'RMSE_LightGBM'] = rmse[1]
        df.loc[i, 'MRE_LightGBM'] = mre[1]
    meb.set_stadata_coords(df, level=0, time=datetime.datetime(2023, 1, 1), dtime=0)
    meb.tool.plot_tools.scatter_sta(df.loc[:, ('level', 'time', 'dtime', 'id', 'lon', 'lat', 'R_EC')],
                                    value_column=0, map_extend=[108.65, 114.4, 24.5, 30.25], cmap=meb.cmaps.ts,
                                    save_path=os.path.join(output_path, 'R_EC.png'), dpi=300)
    meb.tool.plot_tools.scatter_sta(df.loc[:, ('level', 'time', 'dtime', 'id', 'lon', 'lat', 'ME_EC')],
                                    value_column=0, map_extend=[108.65, 114.4, 24.5, 30.25], cmap=meb.cmaps.me,
                                    save_path=os.path.join(output_path, 'ME_EC.png'), dpi=300)
    meb.tool.plot_tools.scatter_sta(df.loc[:, ('level', 'time', 'dtime', 'id', 'lon', 'lat', 'MAE_EC')],
                                    value_column=0, map_extend=[108.65, 114.4, 24.5, 30.25], cmap=meb.cmaps.mae,
                                    save_path=os.path.join(output_path, 'MAE_EC.png'), dpi=300)
    meb.tool.plot_tools.scatter_sta(df.loc[:, ('level', 'time', 'dtime', 'id', 'lon', 'lat', 'RMSE_EC')],
                                    value_column=0, map_extend=[108.65, 114.4, 24.5, 30.25], cmap=meb.cmaps.mae,
                                    save_path=os.path.join(output_path, 'RMSE_EC.png'), dpi=300)
    meb.tool.plot_tools.scatter_sta(df.loc[:, ('level', 'time', 'dtime', 'id', 'lon', 'lat', 'MRE_EC')],
                                    value_column=0, map_extend=[108.65, 114.4, 24.5, 30.25], cmap=meb.cmaps.ts,
                                    save_path=os.path.join(output_path, 'MRE_EC.png'), dpi=300)
    meb.tool.plot_tools.scatter_sta(df.loc[:, ('level', 'time', 'dtime', 'id', 'lon', 'lat', 'R_LightGBM')],
                                    value_column=0, map_extend=[108.65, 114.4, 24.5, 30.25], cmap=meb.cmaps.ts,
                                    save_path=os.path.join(output_path, 'R_LightGBM.png'), dpi=300)
    meb.tool.plot_tools.scatter_sta(df.loc[:, ('level', 'time', 'dtime', 'id', 'lon', 'lat', 'ME_LightGBM')],
                                    value_column=0, map_extend=[108.65, 114.4, 24.5, 30.25], cmap=meb.cmaps.me,
                                    save_path=os.path.join(output_path, 'ME_LightGBM.png'), dpi=300)
    meb.tool.plot_tools.scatter_sta(df.loc[:, ('level', 'time', 'dtime', 'id', 'lon', 'lat', 'MAE_LightGBM')],
                                    value_column=0, map_extend=[108.65, 114.4, 24.5, 30.25], cmap=meb.cmaps.mae,
                                    save_path=os.path.join(output_path, 'MAE_LightGBM.png'), dpi=300)
    meb.tool.plot_tools.scatter_sta(df.loc[:, ('level', 'time', 'dtime', 'id', 'lon', 'lat', 'RMSE_LightGBM')],
                                    value_column=0, map_extend=[108.65, 114.4, 24.5, 30.25], cmap=meb.cmaps.mae,
                                    save_path=os.path.join(output_path, 'RMSE_LightGBM.png'), dpi=300)
    meb.tool.plot_tools.scatter_sta(df.loc[:, ('level', 'time', 'dtime', 'id', 'lon', 'lat', 'MRE_LightGBM')],
                                    value_column=0, map_extend=[108.65, 114.4, 24.5, 30.25], cmap=meb.cmaps.ts,
                                    save_path=os.path.join(output_path, 'MRE_LightGBM.png'), dpi=300)
    df.to_csv(os.path.join(output_path, '分站点.csv'), encoding='gb2312', index=False)


if __name__ == '__main__':
    print('The program "access.py" is beginning.')
    start = arrow.now()

    if len(sys.argv) == 1:
        main(r'D:\output\wind5', r'D:\model\wind5\图',
             r'D:\Project\wind\国家气象观测站.csv')
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2], sys.argv[3])

    end = arrow.now()
    running_time = (end - start).total_seconds()

    print('The program "access.py" runs out in {:s}.'.format(format_time(running_time)))
