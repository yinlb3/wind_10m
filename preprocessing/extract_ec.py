#!user/bin.python3
# -*- coding: utf-8 -*-

"""
ECMWF-IFS 数据提取模块

从 EC4AI 二进制文件中提取气象要素到指定站点, 输出CSV格式数据。

Founded in 2023-07-15
Modified in 2026-04-03
@author: yinlb
"""

import sys
from pathlib import Path

import arrow
import numpy as np
import pandas as pd
from meteva import base as meb

from ..src.utils import format_time


# 提取的EC要素列表（30个）
# 地面要素：2T, MSL, 2D, 10U, 10V
# 各层要素（1000/950/925/900/850 hPa）：T, GH, R, U, V
ELEMENTS = (
    # 地面要素
    '2T', 'MSL', '2D', '10U', '10V',
    # 1000hPa
    'T_1000', 'GH_1000', 'R_1000', 'U_1000', 'V_1000',
    # 950hPa
    'T_950', 'GH_950', 'R_950', 'U_950', 'V_950',
    # 925hPa
    'T_925', 'GH_925', 'R_925', 'U_925', 'V_925',
    # 900hPa
    'T_900', 'GH_900', 'R_900', 'U_900', 'V_900',
    # 850hPa
    'T_850', 'GH_850', 'R_850', 'U_850', 'V_850'
)


def main(input_path: str, output_path: str, station_information_path: str) -> None:
    """
    主函数: 从 EC4AI 提取数据到站点
    
    遍历指定时间范围的 EC 预报数据, 提取30个要素到97个站点, 输出CSV文件。
    
    参数:
        input_path: EC4AI 二进制文件根目录
        output_path: 输出CSV文件目录
        station_information_path: 站点信息CSV文件路径 (包含台站号、经度、纬度)
    
    输出:
        按时间命名的CSV文件, 格式为 YYYYMMDDHH.{dtime}.csv
    """
    # 读取站点信息
    station_information = pd.read_csv(
        station_information_path,
        encoding='gb2312',
        low_memory=False
    )
    station = meb.sta_data(
        station_information.loc[:, ('台站号', '经度', '纬度')],
        columns=('id', 'lon', 'lat')
    )
    # 定义EC数据网格（0.125°和0.25°两种分辨率）
    grid0 = meb.grid((70, 140, 0.125), (0, 60, 0.125))
    grid1 = meb.grid((70, 140, 0.25), (0, 60, 0.25))

    input_dir = Path(input_path)
    output_dir = Path(output_path)
    # 遍历时间范围，读取数据并输出CSV
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
                file_path = input_dir / yyyymm / file / filename
                if not file_path.exists():
                    sta = None
                    break
                data = np.fromfile(str(file_path), dtype=np.float32)
                if i < 5:
                    grd = meb.grid_data(grid0, data)
                else:
                    grd = meb.grid_data(grid1, data)
                sta0 = meb.interp_gs_nearest(grd, station)
                meb.set_stadata_names(sta0, e)
                meb.set_stadata_coords(
                    sta0, level=0, time=time_shift.datetime, dtime=dtime
                )
                sta = meb.combine_on_id(sta, sta0)
            if sta is not None:
                filename = '{:s}.{:03d}.csv'.format(
                    time_shift.format('YYYYMMDDHH'), dtime
                )
                sta.to_csv(output_dir / filename, index=False)
        time_shift = time_shift.shift(hours=12)


if __name__ == '__main__':
    print('The program "extract_ec.py" is beginning.')
    start = arrow.now()

    if len(sys.argv) == 1:
        main(
            input_path=r'\\10.110.173.91\sqxt\EC4AI',
            output_path=r'D:\data\ec_station\wind97',
            station_information_path=r'D:\Project\wind\国家气象观测站.csv'
        )
    elif len(sys.argv) == 4:
        main(
            input_path=sys.argv[1],
            output_path=sys.argv[2],
            station_information_path=sys.argv[3]
        )

    end = arrow.now()
    running_time = (end - start).total_seconds()

    print('The program "extract_ec.py" runs out in {:s}.'.format(format_time(running_time)))
