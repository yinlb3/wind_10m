#!user/bin.python3

"""
Founded in 2024-07-27
Modified in 2024-07-27
@author: yinlb
"""
import typing

import joblib
import os
import sys

import arrow
import lightgbm as lgbm
import numpy as np
import pandas as pd


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
    return data


def uv_to_ds(uv: np.ndarray) -> np.ndarray:
    ds = np.zeros_like(uv, dtype=np.float32) + np.nan
    ds[:, :, :, 1] = np.sqrt(uv[:, :, :, 0] ** 2 + uv[:, :, :, 1] ** 2)
    ds[:, :, :, 0] = 180 + np.arctan2(uv[:, :, :, 0], uv[:, :, :, 1]) * 180 / np.pi
    return ds


def train(data: np.ndarray, label: np.ndarray) -> lgbm.LGBMRegressor:
    index = ~np.isnan(label)
    for i in range(data.shape[-1]):
        index &= ~np.isnan(data[:, :, :, i])
    x = data[index, :]
    y = label[index]
    model = lgbm.LGBMRegressor(n_jobs=20, verbose=-1, force_col_wise=True, random_state=RANDOM_SEED)
    model.fit(x, y)
    return model


def train2(data: np.ndarray, label: np.ndarray, option: int) -> typing.List:
    models = list()
    a, b, c, d = data.shape
    if option == 1:
        models.append(train(data, label))
    elif option == 2:
        for i in range(b):
            temp_data = np.reshape(data[:, i, :, :], newshape=(a, 1, c, d))
            temp_label = np.reshape(label[:, i, :], newshape=(a, 1, c))
            models.append(train(temp_data, temp_label))
    elif option == 3:
        for i in range(c):
            temp_data = np.reshape(data[:, :, i, :], newshape=(a, b, 1, d))
            temp_label = np.reshape(label[:, :, i], newshape=(a, b, 1))
            models.append(train(temp_data, temp_label))
    elif option == 4:
        for i in range(b):
            models.append(list())
            for j in range(c):
                temp_data = np.reshape(data[:, i, j, :], newshape=(a, 1, 1, d))
                temp_label = np.reshape(label[:, i, j], newshape=(a, 1, 1))
                models[i].append(train(temp_data, temp_label))
    return models


def test(model: lgbm.LGBMRegressor, data: np.ndarray) -> np.ndarray:
    index = np.ones_like(data[:, :, :, 0], dtype=np.bool_)
    for i in range(data.shape[-1]):
        index &= ~np.isnan(data[:, :, :, i])
    label = np.zeros_like(data[:, :, :, 0], dtype=np.float32) + np.nan
    x = data[index, :]
    label[index] = model.predict(x)
    return label


def test2(models: typing.List, data: np.ndarray, option: int) -> np.ndarray:
    a, b, c, d = data.shape
    label = np.zeros(shape=(a, b, c), dtype=data.dtype) + np.nan
    if option == 1:
        label = test(models[0], data)
    elif option == 2:
        for i in range(b):
            model = models[i]
            temp_data = np.reshape(data[:, i, :, :], newshape=(a, 1, c, d))
            temp_label = test(model, temp_data)
            label[:, i, :] = np.reshape(temp_label, newshape=(a, c))
    elif option == 3:
        for i in range(c):
            model = models[i]
            temp_data = np.reshape(data[:, :, i, :], newshape=(a, b, 1, d))
            temp_label = test(model, temp_data)
            label[:, :, i] = np.reshape(temp_label, newshape=(a, b))
    elif option == 4:
        for i in range(b):
            for j in range(c):
                model = models[i][j]
                temp_data = np.reshape(data[:, i, j, :], newshape=(a, 1, 1, d))
                temp_label = test(model, temp_data)
                label[:, i, j] = np.reshape(temp_label, newshape=a)
    return label


def test3(models: typing.List, data: np.ndarray, option: int) -> np.ndarray:
    a, b, c, d = data.shape
    label = np.zeros(shape=(c, a, b, c), dtype=data.dtype) + np.nan
    if option == 3:
        for i in range(c):
            model = models[i]
            for j in range(c):
                temp_data = np.reshape(data[:, :, j, :], newshape=(a, b, 1, d))
                temp_label = test(model, temp_data)
                label[i, :, :, j] = np.reshape(temp_label, newshape=(a, b))
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
        print(i)

    for i in range(4, 6):
        year = 2017 + i
        data_list2 = data_list.copy()
        data_list2.pop(i)
        label_list2 = label_list.copy()
        label_list2.pop(i)
        data = np.vstack(data_list2)
        train_data = data
        train_data = interp(train_data[:, 12:37, :, :])[:, 1:, :, :]
        train_label = np.vstack(label_list2)
        train_label = train_label[:, 13:37, :, 3]
        train_label[train_label > 100] = np.nan
        data = data_list[i]
        test_data = data
        test_data = interp(test_data[:, 12:37, :, :])[:, 1:, :, :]
        test_label = label_list[i]
        test_label = test_label[:, 13:37, :, 3]
        test_label[test_label > 100] = np.nan
        nwp_train_label = uv_to_ds(train_data[:, :, :, 3:5])[:, :, :, 1]
        nwp_test_label = uv_to_ds(test_data[:, :, :, 3:5])[:, :, :, 1]
        models = joblib.load(rf'E:\123\model\wind5\lgb_{year}.joblib')
        # joblib.dump(models, os.path.join(model_path, 'lgb_{:d}.joblib'.format(year)))
        # predict_label = test(model, test_data)
        # predict_label = nwp_test_label - test(model, test_data)
        predict_label = test3(models, test_data, 3)
        np.save(os.path.join(output_path, f'lgb_3_{year}_cross.npy'), predict_label)


if __name__ == '__main__':
    print('The program "LightGBM.py" is beginning.')
    start = arrow.now()

    if len(sys.argv) == 1:
        main(r'D:\data\wind', r'D:\model\wind5', r'D:\Project\wind\97',
             r'D:\Project\wind\国家气象观测站.csv')
    elif len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

    end = arrow.now()
    running_time = (end - start).total_seconds()

    print('The program "LightGBM.py" runs out in {:s}.'.format(format_time(running_time)))
