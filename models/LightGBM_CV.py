#!user/bin.python3

"""
Founded in 2023-08-01
Modified in 2023-08-01
@author: yinlb
"""
import os
import sys

import arrow
import lightgbm as lgb
import numpy as np
from sklearn import model_selection as ms


RANDOM_SEED = 20230718


def interp(data: np.ndarray) -> np.ndarray:
    for i in range(8):
        data[:, 3 * i + 1, :, :] = data[:, 3 * i, :, :] * 2 / 3 + data[:, 3 * i + 3, :, :] / 3
        data[:, 3 * i + 2, :, :] = data[:, 3 * i, :, :] / 3 + data[:, 3 * i + 3, :, :] * 2 / 3
    return data[:, 1:, :, :]


def clean(data: np.ndarray) -> np.ndarray:
    no_wind = (data[:, 1:, :, 0] == 999017) & (data[:, 1:, :, 1] == 0)
    index0 = data[:, 1:, :, 0] <= 360
    index1 = data[:, 1:, :, 1] < 999017
    delta = data[:, 1:, :, 1] - data[:, :-1, :, 1]
    index1 &= np.abs(delta) < 10
    index0 |= no_wind
    index = index0 & index1
    data = data[:, 1:, :, :]
    data[~index, :] = np.nan
    return data


def ds_to_uv(ds: np.ndarray) -> np.ndarray:
    no_wind = ds[:, :, :, 0] == 999017
    uv = np.zeros_like(ds) + np.nan
    uv[~no_wind, 0] = -ds[~no_wind, 1] * np.sin(np.deg2rad(ds[~no_wind, 0]))
    uv[~no_wind, 1] = -ds[~no_wind, 1] * np.cos(np.deg2rad(ds[~no_wind, 0]))
    uv[no_wind, :] = 0
    return uv


def uv_to_ds(uv: np.ndarray) -> np.ndarray:
    ds = np.zeros_like(uv) + np.nan
    ds[:, :, :, 1] = np.sqrt(uv[:, :, :, 0] ** 2 + uv[:, :, :, 1] ** 2)
    ds[:, :, :, 0] = 180 + np.arctan2(uv[:, :, :, 0], uv[:, :, :, 1]) * 180 / np.pi
    return ds


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


def main(input_path: str):
    data = list()
    label = list()
    for i in range(2017, 2023):
        data.append(np.load(os.path.join(input_path, 'nwp{:d}.npy'.format(i))))
        label.append(np.load(os.path.join(input_path, 'ob{:d}.npy'.format(i))))
        print(i)
    data = np.vstack(data)
    data = data[:, 12:37, :, :]
    data = interp(data)
    label = np.vstack(label)
    label = label[:, 12:37, :, 2:4]
    label = clean(label)
    uv = ds_to_uv(label)

    index = ~np.isnan(uv[:, :, :, 0])
    for i in range(data.shape[-1]):
        index &= ~np.isnan(data[:, :, :, i])
    x = data[index, :]
    y = uv[:, :, :, 0][index]
    data_mean = np.mean(x, axis=0)
    data_std = np.std(x, axis=0)
    x = (x - data_mean) / data_std
    model = lgb.LGBMRegressor(n_jobs=20, verbose=0, force_col_wise=True, random_state=RANDOM_SEED)

    parameters = {
        # 'learning_rate': (0.05, 0.1),
        # 'n_estimators': (100, 500, 1000),
        # 'min_child_samples': (20, 30),
        # 'min_child_weight': (0.001, 1),
        'max_depth': (3, 4, 5),
        'num_leaves': (7, 15, 31),
        'subsample': (0.8, 0.9, 1.0),
        'colsample_bytree': (0.8, 0.9, 1.0),
        'reg_alpha': (1, 10, 1000),
        'reg_lambda': (1, 10, 1000)
    }
    clf = ms.RandomizedSearchCV(model, parameters, verbose=2)
    clf.fit(x, y)
    print(clf)


if __name__ == '__main__':
    print('The program "LightGBM_CV.py" is beginning.')
    start = arrow.now()

    if len(sys.argv) == 1:
        main(r'D:\data\wind')
    elif len(sys.argv) == 5:
        main(sys.argv[1])

    end = arrow.now()
    running_time = (end - start).total_seconds()

    print('The program "LightGBM_CV.py" runs out in {:s}.'.format(format_time(running_time)))
