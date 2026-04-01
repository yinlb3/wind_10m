#!user/bin.python3

"""
Founded in 2023-07-18
Modified in 2024-10-11
@author: yinlb
"""
import joblib
import os
import sys
import typing

import arrow
import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn import discriminant_analysis as da
from sklearn import linear_model as lm


RANDOM_SEED = 20230718


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
    return data[:, 1:, :, :]


def clean(data: np.ndarray) -> np.ndarray:
    no_wind = (data[:, 1:, :, 0] == 999017) & (data[:, 1:, :, 1] < 0.3)
    index0 = data[:, 1:, :, 0] <= 360
    index1 = data[:, 1:, :, 1] < 999017
    delta = data[:, 1:, :, 1] - data[:, :-1, :, 1]
    index1 &= np.abs(delta) < 10
    index0 |= no_wind
    index = index0 & index1
    data = data[:, 1:, :, :]
    data[~index, :] = np.nan
    data[no_wind, 0] = 999017
    return data


def ds_to_uv(ds: np.ndarray) -> np.ndarray:
    no_wind = ds[:, :, :, 0] == 999017
    uv = np.zeros_like(ds, dtype=np.float32) + np.nan
    uv[~no_wind, 0] = -ds[~no_wind, 1] * np.sin(np.deg2rad(ds[~no_wind, 0]))
    uv[~no_wind, 1] = -ds[~no_wind, 1] * np.cos(np.deg2rad(ds[~no_wind, 0]))
    uv[no_wind, 0] = 999017
    uv[no_wind, 1] = ds[no_wind, 1]
    return uv


def uv_to_ds(uv: np.ndarray) -> np.ndarray:
    ds = np.zeros_like(uv, dtype=np.float32) + np.nan
    no_wind = uv[:, :, :, 0] == 999017
    wind_speed = np.sqrt(uv[no_wind, 0] ** 2 + uv[no_wind, 1] ** 2)
    wind_speed[wind_speed >= 0.3] = 0.29
    ds[no_wind, 1] = wind_speed
    ds[no_wind, 0] = 999017
    wind_speed = np.sqrt(uv[~no_wind, 0] ** 2 + uv[~no_wind, 1] ** 2)
    wind_speed[wind_speed < 0.3] = 0.3
    ds[~no_wind, 1] = wind_speed
    ds[~no_wind, 0] = 180 + np.arctan2(uv[~no_wind, 0], uv[~no_wind, 1]) * 180 / np.pi
    return ds


def train(data: np.ndarray, label: np.ndarray, model_name: str):
    assert model_name in ('lgb', 'mlr')
    index = ~np.isnan(label[:, :, :, 0])
    index &= ~np.isnan(label[:, :, :, 1])
    for i in range(data.shape[-1]):
        index &= ~np.isnan(data[:, :, :, i])
    x = data[index, :]
    y = label[index, :]
    models = list()
    index = y[:, 0] == 999017
    # if np.sum(index) >= 2:
    #     y_ = np.zeros_like(y[:, 0])
    #     y_[~index] = 1
    #     if model_name == 'lgb':
    #         model = lgbm.LGBMClassifier(n_jobs=20, verbose=-1, force_col_wise=True, random_state=RANDOM_SEED)
    #     elif model_name == 'mlr':
    #         model = da.LinearDiscriminantAnalysis()
    #     model.fit(x, y_)
    #     models.append(model)
    #     y_ = y[index, 1]
    #     if model_name == 'lgb':
    #         model = lgbm.LGBMRegressor(n_jobs=20, verbose=-1, force_col_wise=True, random_state=RANDOM_SEED)
    #     elif model_name == 'mlr':
    #         model = lm.LinearRegression()
    #     model.fit(x[index], y_)
    #     models.append(model)
    # else:
    #     models.append(1)
    #     models.append(1)
    y_ = y[~index, 0]
    if model_name == 'lgb':
        model = lgbm.LGBMRegressor(n_jobs=20, verbose=-1, force_col_wise=True, random_state=RANDOM_SEED)
    elif model_name == 'mlr':
        model = lm.LinearRegression()
    model.fit(x[~index], y_)
    models.append(model)
    y_ = y[~index, 1]
    if model_name == 'lgb':
        model = lgbm.LGBMRegressor(n_jobs=20, verbose=-1, force_col_wise=True, random_state=RANDOM_SEED)
    elif model_name == 'mlr':
        model = lm.LinearRegression()
    model.fit(x[~index], y_)
    models.append(model)
    return models


def train2(data: np.ndarray, label: np.ndarray, option: int, model_name: str) -> typing.List:
    models = list()
    a, b, c, d = data.shape
    if option == 1:
        models = train(data, label, model_name)
    elif option == 2:
        for i in range(b):
            temp_data = np.reshape(data[:, i, :, :], newshape=(a, 1, c, d))
            temp_label = np.reshape(label[:, i, :, :], newshape=(a, 1, c, 2))
            models.append(train(temp_data, temp_label, model_name))
    elif option == 3:
        for i in range(c):
            temp_data = np.reshape(data[:, :, i, :], newshape=(a, b, 1, d))
            temp_label = np.reshape(label[:, :, i, :], newshape=(a, b, 1, 2))
            models.append(train(temp_data, temp_label, model_name))
    elif option == 4:
        for i in range(b):
            models.append(list())
            for j in range(c):
                temp_data = np.reshape(data[:, i, j, :], newshape=(a, 1, 1, d))
                temp_label = np.reshape(label[:, i, j, :], newshape=(a, 1, 1, 2))
                models[i].append(train(temp_data, temp_label, model_name))
    return models


def test(models: typing.List, data: np.ndarray) -> np.ndarray:
    index = np.ones(data.shape[:-1], dtype=np.bool_)
    for i in range(data.shape[-1]):
        index &= ~np.isnan(data[:, ..., i])
    label = np.zeros(list(data.shape[:-1]) + [2], dtype=np.float32) + np.nan
    x = data[index, :]
    # u = models[2].predict(x)
    # v = models[3].predict(x)
    # uv = np.stack([u, v], axis=1)
    # if models[0] != 1 and models[1] != 1:
    #     is_wind = models[0].predict(x)
    #     if np.sum(is_wind == 0) > 0:
    #         no_wind_speed = models[1].predict(x[is_wind == 0, :])
    #         uv[is_wind == 0, 0] = 999017
    #         uv[is_wind == 0, 1] = no_wind_speed
    u = models[0].predict(x)
    v = models[1].predict(x)
    uv = np.stack([u, v], axis=1)
    label[index] = uv
    return label


def test2(models: typing.List, data: np.ndarray, option: int) -> np.ndarray:
    a, b, c, d = data.shape
    label = np.zeros((a, b, c, 2), dtype=np.float32) + np.nan
    if option == 1:
        label = test(models, data)
    elif option == 2:
        for i in range(b):
            label[:, i, :, :] = test(models[i], data[:, i, :, :])
    elif option == 3:
        for i in range(c):
            label[:, :, i, :] = test(models[i], data[:, :, i, :])
    elif option == 4:
        for i in range(b):
            for j in range(c):
                label[:, i, j, :] = test(models[i][j], data[:, i, j, :])
    return label


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


def main(input_path: str, model_path: str, output_path: str, station_information_path: str):
    station_information = pd.read_csv(station_information_path, encoding='gb2312', low_memory=False)
    station_information.sort_values(by=['台站号'], inplace=True)
    station_information.reset_index(drop=True, inplace=True)

    data_list = list()
    label_list = list()
    for i in range(2017, 2023):
        data_list.append(np.load(os.path.join(input_path, 'nwp{:d}.npy'.format(i))))
        label_list.append(np.load(os.path.join(input_path, 'ob{:d}.npy'.format(i))))

    lgb = {1: list(), 2: list(), 3: list(), 4: list()}
    mlr = {1: list(), 2: list(), 3: list(), 4: list()}
    for i in range(6):
        year = 2017 + i
        data_list2 = data_list.copy()
        data_list2.pop(i)
        label_list2 = label_list.copy()
        label_list2.pop(i)
        train_data = np.vstack(data_list2)[:, 12:37, :, :]
        train_data = interp(train_data)
        train_label = np.vstack(label_list2)
        train_label = train_label[:, 12:37, :, 2:4]
        train_label = clean(train_label)
        train_uv = ds_to_uv(train_label)
        test_data = data_list[i][:, 12:37, :, :]
        test_data = interp(test_data)
        test_label = label_list[i]
        test_label = test_label[:, 12:37, :, 2:4]
        test_label = clean(test_label)
        np.save(os.path.join(output_path, f'ob_{year}.npy'), test_label)
        nwp_uv = test_data[:, :, :, 3:5]
        nwp_label = uv_to_ds(nwp_uv)
        np.save(os.path.join(output_path, f'nwp_{year}.npy'), nwp_label)

        for j in range(4):
            time_arrow = arrow.now()
            models = train2(train_data, train_uv, j + 1, 'lgb')
            lgb[j + 1].append((arrow.now() - time_arrow).total_seconds() / 60)
            predict_uv = test2(models, test_data, j + 1)
            predict_label = uv_to_ds(predict_uv)
            np.save(os.path.join(output_path, f'lgb_{j + 1}_{year}.npy'), predict_label)
            train_x = np.copy(train_data)
            test_x = np.copy(test_data)
            a, b, c, d = train_x.shape
            if j == 0:
                data_mean = np.nanmean(train_data, axis=(0, 1, 2))
                data_std = np.nanstd(train_data, axis=(0, 1, 2))
                train_x = (train_x - data_mean) / data_std
                test_x = (test_x - data_mean) / data_std
            elif j == 1:
                for k in range(b):
                    data_mean = np.nanmean(train_data[:, k, :, :], axis=(0, 1))
                    data_std = np.nanstd(train_data[:, k, :, :], axis=(0, 1))
                    train_x[:, k, :, :] = (train_x[:, k, :, :] - data_mean) / data_std
                    test_x[:, k, :, :] = (test_x[:, k, :, :] - data_mean) / data_std
            elif j == 2:
                for k in range(c):
                    data_mean = np.nanmean(train_data[:, :, k, :], axis=(0, 1))
                    data_std = np.nanstd(train_data[:, :, k, :], axis=(0, 1))
                    train_x[:, :, k, :] = (train_x[:, :, k, :] - data_mean) / data_std
                    test_x[:, :, k, :] = (test_x[:, :, k, :] - data_mean) / data_std
            elif j == 3:
                for k in range(b):
                    for ii in range(c):
                        data_mean = np.nanmean(train_data[:, k, ii, :], axis=0)
                        data_std = np.nanstd(train_data[:, k, ii, :], axis=0)
                        train_x[:, k, ii, :] = (train_x[:, k, ii, :] - data_mean) / data_std
                        test_x[:, k, ii, :] = (test_x[:, k, ii, :] - data_mean) / data_std
            time_arrow = arrow.now()
            models = train2(train_x, train_uv, j + 1, 'mlr')
            mlr[j + 1].append((arrow.now() - time_arrow).total_seconds() / 60)
            predict_uv = test2(models, test_x, j + 1)
            predict_label = uv_to_ds(predict_uv)
            np.save(os.path.join(output_path, f'mlr_{j + 1}_{year}.npy'), predict_label)
        print(year)

    pd.DataFrame(lgb).to_csv(fr'{output_path}/lgb_time.csv', index=False)
    pd.DataFrame(mlr).to_csv(fr'{output_path}/mlr_time.csv', index=False)


if __name__ == '__main__':
    print('The program "LightGBM.py" is beginning.')
    start = arrow.now()

    if len(sys.argv) == 1:
        main(r'D:\data\wind', r'D:\model\wind2', r'D:\Project\wind\97-uv',
             r'D:\Project\wind\国家气象观测站.csv')
    elif len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

    end = arrow.now()
    running_time = (end - start).total_seconds()

    print('The program "LightGBM.py" runs out in {:s}.'.format(format_time(running_time)))
