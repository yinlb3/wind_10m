#!user/bin.python3

"""
Founded in 2023-07-15
Modified in 2026-04-02
@author: yinlb
"""
import os
import sys

import arrow
import numpy as np
import pandas as pd
from meteva import base as meb


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


def main(input_path: str, output_path: str, station_information_path: str):
    station_information = pd.read_csv(station_information_path, encoding='gb2312', low_memory=False)
    station = meb.sta_data(station_information.loc[:, ('台站号', '经度', '纬度')], columns=('id', 'lon', 'lat'))
    grid0 = meb.grid((70, 140, 0.125), (0, 60, 0.125))
    grid1 = meb.grid((70, 140, 0.25), (0, 60, 0.25))

    time_begin = arrow.get('2017')
    time_end = arrow.get('2023')
    time_shift = time_begin
    while time_shift < time_end:
        yyyymm = time_shift.format('YYYYMM')
        for dtime in range(12, 37, 3):
            mmddhh0 = time_shift.format('MMDDHH')
            mmddhh1 = time_shift.shift(hours=dtime).format('MMDDHH')
            file = 'C1D{:s}00{:s}001'.format(mmddhh0, mmddhh1)
            sta = None
            for i, e in enumerate(ELEMENTS):
                filename = '{:s}_{:s}.AI.bin'.format(file, e)
                file_path = os.path.join(input_path, yyyymm, file, filename)
                if not os.path.exists(file_path):
                    sta = None
                    break
                data = np.fromfile(file_path, dtype=np.float32)
                if i < 5:
                    grd = meb.grid_data(grid0, data)
                else:
                    grd = meb.grid_data(grid1, data)
                sta0 = meb.interp_gs_nearest(grd, station)
                meb.set_stadata_names(sta0, e)
                meb.set_stadata_coords(sta0, level=0, time=time_shift.datetime, dtime=dtime)
                sta = meb.combine_on_id(sta, sta0)
            if sta is not None:
                filename = '{:s}.{:03d}.csv'.format(time_shift.format('YYYYMMDDHH'), dtime)
                sta.to_csv(os.path.join(output_path, filename), index=False)
        time_shift = time_shift.shift(hours=12)


if __name__ == '__main__':
    print('The program "extract_ec.py" is beginning.')
    start = arrow.now()

    if len(sys.argv) == 1:
        main(r'\\10.110.173.91\sqxt\EC4AI', r'D:\data\ec_station\wind97', r'D:\Project\wind\国家气象观测站.csv')
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])

    end = arrow.now()
    running_time = (end - start).total_seconds()

    print('The program "extract_ec.py" runs out in {:s}.'.format(format_time(running_time)))
