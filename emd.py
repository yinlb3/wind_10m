#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMD demo script.

Demonstrates Empirical Mode Decomposition on a sample time series.

Founded in 2023-07
Modified in 2026-09-17
@author: yinlb, deepseek-flash
"""

import pathlib

import arrow
import numpy as np
import pandas as pd
import PyEMD
from matplotlib import pyplot as plt

from src import utils


def main() -> None:
    """
    Run EMD decomposition on a sample CSV series and plot the IMFs.

    Reads the configured sample CSV, extracts the target column, applies
    Empirical Mode Decomposition, and visualizes the original signal plus
    each intrinsic mode function (IMF).
    """
    # 1. Load configuration and sample file path.
    cfg = utils.load_config()
    file_path = pathlib.Path(cfg['paths']['emd_sample_path'])

    # 2. Read the CSV file and prepare the signal and time axis.
    data_frame = pd.read_csv(
        filepath_or_buffer=file_path, low_memory=False
    )
    time_index = np.array(data_frame.index)
    col_name = 't2'
    signal = np.array(data_frame.loc[:, col_name])

    # 3. Perform Empirical Mode Decomposition.
    emd_model = PyEMD.EMD()
    imfs = emd_model(signal, time_index)
    num_subplots = imfs.shape[0] + 1

    # 4. Plot the original signal and each IMF.
    plt.subplot(num_subplots, 1, 1)
    plt.plot(time_index, signal, color='red')
    plt.title(col_name)
    plt.xlabel('Time [s]')
    for i, imf in enumerate(imfs):
        plt.subplot(num_subplots, 1, i + 2)
        plt.plot(time_index, imf, color='green')
        plt.title('IMF ' + str(i + 1))
        plt.xlabel('Time [s]')
    plt.show()


if __name__ == '__main__':
    print('Program emd.py started')
    start = arrow.now()

    main()

    elapsed = (arrow.now() - start).total_seconds()
    print(
        'Program emd.py finished, total time {}'.format(
            utils.format_time(elapsed)
        )
    )
