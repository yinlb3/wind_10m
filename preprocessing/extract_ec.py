#!user/bin.python3
# -*- coding: utf-8 -*-

"""
ECMWF-IFS 数据提取模块

从 EC4AI 二进制文件中提取气象要素到指定站点,
输出CSV格式数据。

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


def main(
    input_path: str,
    output_path: str,
    station_info_path: str
) -> None:
    """
    主函数: 从 EC4AI 提取数据到站点
    
    遍历指定时间范围的 EC 预报数据,
    提取30个要素到97个站点, 输出CSV文件。
    
    参数:
        input_path: EC4AI 二进制文件根目录
        output_path: 输出CSV文件目录
        station_info_path: 站点信息CSV文件路径
            (包含台站号、经度、纬度)
    
    输出:
        按时间命名的CSV文件, 格式为 YYYYMMDDHH.{lead_time}.csv
    """
    # 读取站点信息
    station_df = pd.read_csv(
        station_info_path,
        encoding='gb2312',
        low_memory=False
    )
    station_metadata = meb.sta_data(
        station_df.loc[:, ('台站号', '经度', '纬度')],
        columns=('id', 'lon', 'lat')
    )
    # 定义EC数据网格（0.125°高分辨率和0.25°标准分辨率）
    grid_high_res = meb.grid((70, 140, 0.125), (0, 60, 0.125))
    grid_standard_res = meb.grid((70, 140, 0.25), (0, 60, 0.25))
    # 定义输入输出路径
    input_dir = Path(input_path)
    output_dir = Path(output_path)
    # 遍历时间范围，读取数据并输出CSV
    time_begin = arrow.get('2017')
    time_end = arrow.get('2023')
    forecast_reference_time = time_begin
    while forecast_reference_time < time_end:
        year_month = forecast_reference_time.format('YYYYMM')
        for lead_time in range(12, 37, 3):
            init_time = forecast_reference_time.format('MMDDHH')
            valid_time = forecast_reference_time.shift(
                hours=lead_time
            ).format('MMDDHH')
            ec_filename_base = 'C1D{:s}00{:s}001'.format(init_time, valid_time)
            station_combined = None
            for elem_idx, element in enumerate(ELEMENTS):
                filename = '{:s}_{:s}.AI.bin'.format(ec_filename_base, element)
                file_path = input_dir / year_month / ec_filename_base / filename
                if not file_path.exists():
                    station_combined = None
                    break
                raw_data = np.fromfile(str(file_path), dtype=np.float32)
                # 前5个要素使用0.125°高分辨率网格
                if elem_idx < 5:
                    grid_data = meb.grid_data(grid_high_res, raw_data)
                else:
                    grid_data = meb.grid_data(grid_standard_res, raw_data)
                station_single = meb.interp_gs_nearest(
                    grid_data, station_metadata
                )
                meb.set_stadata_names(station_single, element)
                meb.set_stadata_coords(
                    station_single,
                    level=0,
                    time=forecast_reference_time.datetime,
                    dtime=lead_time
                )
                station_combined = meb.combine_on_id(
                    station_combined, station_single
                )
            if station_combined is not None:
                output_filename = '{:s}.{:03d}.csv'.format(
                    forecast_reference_time.format('YYYYMMDDHH'), lead_time
                )
                station_combined.to_csv(
                    output_dir / output_filename, index=False
                )
        forecast_reference_time = forecast_reference_time.shift(hours=12)


if __name__ == '__main__':
    msg = 'The program "extract_ec.py" is beginning.'
    print(msg)
    start = arrow.now()

    if len(sys.argv) == 1:
        main(
            input_path=r'\\10.110.173.91\sqxt\EC4AI',
            output_path=r'D:\data\ec_station\wind97',
            station_info_path=r'D:\Project\wind\国家气象观测站.csv'
        )
    elif len(sys.argv) == 4:
        main(
            input_path=sys.argv[1],
            output_path=sys.argv[2],
            station_info_path=sys.argv[3]
        )

    end = arrow.now()
    running_time = (end - start).total_seconds()
    time_str = format_time(running_time)
    print('The program "extract_ec.py" runs out in {:s}.'.format(time_str))
