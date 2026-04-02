#!user/bin.python3
# -*- coding: utf-8 -*-

"""
数据格式转换模块

将EC预报CSV和观测数据CSV转换为numpy数组格式 (.npy), 便于后续模型训练。

Founded in 2023-07-17
Modified in 2026-04-03
@author: yinlb
"""

import sys
from pathlib import Path

import arrow
import numpy as np
import pandas as pd

from ..src import utils


# EC要素列表（30个），与extract_ec.py保持一致
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
    input_path2: str,
    output_path: str,
    station_information_path: str,
    year: int
) -> None:
    """
    主函数: 转换CSV数据为npy格式
    
    读取EC预报数据和观测数据, 转换为numpy数组并保存, 供模型训练使用。
    
    参数:
        input_path: EC预报CSV文件目录 (由extract_ec.py生成)
        input_path2: 国家站观测数据CSV文件目录
        output_path: 输出npy文件目录
        station_information_path: 站点信息CSV文件路径
        year: 处理年份 (2017-2022)
    
    输出:
        nwp{year}.npy: EC预报数据, shape为 (样本数, 85时效, 97站点, 30要素)
        ob{year}.npy: 观测数据, shape为 (样本数, 85时效, 97站点, 8要素)
    """
    # ---- 1. 初始化：读取站点信息和预分配数组 ----
    station_information = pd.read_csv(
        station_information_path,
        encoding='gb2312',
        low_memory=False
    )
    station_information = station_information.loc[:, '台站号'].to_list()
    station_information.sort()
    time_begin = arrow.get(str(year))
    time_end = time_begin.shift(years=1)
    n = round((time_end - time_begin).total_seconds() / 3600 / 12)
    nwp = np.zeros((n, 85, 97, 30), dtype=np.float32) + np.nan
    ob = np.zeros((n, 85, 97, 8), dtype=np.float32) + np.nan
    print(year, n)

    input_dir = Path(input_path)
    input_dir2 = Path(input_path2)
    output_dir = Path(output_path)
    # ---- 2. 处理EC预报数据（读取CSV并保存npy） ----
    for i in range(n):
        time_shift = time_begin.shift(hours=12 * i)
        for dtime in range(0, 85, 3):
            filename = '{:s}.{:03d}.csv'.format(
                time_shift.format('YYYYMMDDHH'), dtime
            )
            file_path = input_dir / time_shift.format('YYYY') / filename
            if file_path.exists():
                df = pd.read_csv(file_path, low_memory=False)
                df.sort_values(by=['id'], inplace=True)
                df.reset_index(drop=True, inplace=True)
                for j, e in enumerate(ELEMENTS):
                    nwp[i, dtime, :, j] = df.loc[:, e]
    np.save(
        output_dir / 'nwp{:d}.npy'.format(year),
        nwp
    )

    # ---- 3. 处理观测数据（读取CSV并保存npy） ----
    for i in range(n):
        for dtime in range(85):
            time_shift = time_begin.shift(hours=12 * i + dtime)
            filename = time_shift.format('YYYYMMDDHH') + '.csv'
            file_path = input_dir2 / time_shift.format('YYYY') / filename
            if file_path.exists():
                df = pd.read_csv(file_path, low_memory=False)
                # 选择需要的观测要素（8个风要素）
                df = df.loc[:, (
                    '区站号',
                    '2分钟平均风向(角度)',
                    '2分钟平均风速',
                    '10分钟平均风向(角度)',
                    '10分钟平均风速',
                    '小时内最大风速的风向',
                    '最大风速',
                    '极大风速的风向(角度)',
                    '极大风速'
                )]
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
    np.save(
        output_dir / 'ob{:d}.npy'.format(year),
        ob
    )


if __name__ == '__main__':
    print('The program "convert_npy.py" is beginning.')
    start = arrow.now()

    if len(sys.argv) == 1:
        for y in range(2017, 2023):
            main(
                input_path=r'D:\data\ec_station\wind97',
                input_path2=r'D:\data\国家站',
                output_path=r'D:\data\wind',
                station_information_path=r'D:\Project\wind\国家气象观测站.csv',
                year=y
            )
    elif len(sys.argv) == 6:
        main(
            input_path=sys.argv[1],
            input_path2=sys.argv[2],
            output_path=sys.argv[3],
            station_information_path=sys.argv[4],
            year=int(sys.argv[5])
        )

    end = arrow.now()
    running_time = (end - start).total_seconds()

    print('The program "convert_npy.py" runs out in {:s}.'.format(utils.format_time(running_time)))
