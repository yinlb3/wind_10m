#!user/bin.python3

"""
Founded in 2023-07-17
Modified in 2026-04-02
@author: yinlb
"""
import os
import sys

import arrow
import numpy as np
import pandas as pd


ELEMENTS = ('2T', 'MSL', '2D', '10U', '10V', 'T_1000', 'GH_1000', 'R_1000', 'U_1000', 'V_1000', 'T_950', 'GH_950',
            'R_950', 'U_950', 'V_950', 'T_925', 'GH_925', 'R_925', 'U_925', 'V_925', 'T_900', 'GH_900', 'R_900',
            'U_900', 'V_900', 'T_850', 'GH_850', 'R_850', 'U_850', 'V_850')


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


def main(input_path: str, input_path2: str, output_path: str, station_information_path: str, year: int):
    station_information = pd.read_csv(station_information_path, encoding='gb2312', low_memory=False)
    station_information = station_information.loc[:, '台站号'].to_list()
    station_information.sort()

    time_begin = arrow.get(str(year))
    time_end = time_begin.shift(years=1)
    n = round((time_end - time_begin).total_seconds() / 3600 / 12)
    nwp = np.zeros((n, 85, 97, 30), dtype=np.float32) + np.nan
    ob = np.zeros((n, 85, 97, 8), dtype=np.float32) + np.nan
    print(year, n)

    for i in range(n):
        time_shift = time_begin.shift(hours=12 * i)
        for dtime in range(0, 85, 3):
            filename = '{:s}.{:03d}.csv'.format(time_shift.format('YYYYMMDDHH'), dtime)
            file_path = os.path.join(input_path, time_shift.format('YYYY'), filename)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, low_memory=False)
                df.sort_values(by=['id'], inplace=True)
                df.reset_index(drop=True, inplace=True)
                for j, e in enumerate(ELEMENTS):
                    nwp[i, dtime, :, j] = df.loc[:, e]
    np.save(os.path.join(output_path, 'nwp{:d}.npy'.format(year)), nwp)

    for i in range(n):
        for dtime in range(85):
            time_shift = time_begin.shift(hours=12 * i + dtime)
            filename = time_shift.format('YYYYMMDDHH') + '.csv'
            file_path = os.path.join(input_path2, time_shift.format('YYYY'), filename)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, low_memory=False)
                df = df.loc[:, ('区站号', '2分钟平均风向(角度)', '2分钟平均风速', '10分钟平均风向(角度)', '10分钟平均风速',
                                '小时内最大风速的风向', '最大风速', '极大风速的风向(角度)', '极大风速')]
                df.sort_values(by=['区站号'], inplace=True)
                df.reset_index(drop=True, inplace=True)
                for j in range(8):
                    if len(df) == 97:
                        ob[i, dtime, :, j] = df.iloc[:, j + 1]
                    else:
                        for k in range(len(df)):
                            sta_id = df.iloc[k, 0]
                            if sta_id in station_information:
                                ind = station_information.index(sta_id)
                                ob[i, dtime, ind, j] = df.iloc[k, j + 1]
    np.save(os.path.join(output_path, 'ob{:d}.npy'.format(year)), ob)


if __name__ == '__main__':
    print('The program "convert_npy.py" is beginning.')
    start = arrow.now()

    if len(sys.argv) == 1:
        for y in range(2017, 2023):
            main(r'D:\data\ec_station\wind97', r'D:\data\国家站', r'D:\data\wind',
                 r'D:\Project\wind\国家气象观测站.csv', y)
    elif len(sys.argv) == 6:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))

    end = arrow.now()
    running_time = (end - start).total_seconds()

    print('The program "convert_npy.py" runs out in {:s}.'.format(format_time(running_time)))
