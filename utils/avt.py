#!user/bin.python3

"""
Founded in 2024-08-20
Modified in 2024-08-20
@author: yinlb
"""
import os
import sys
import typing

import arrow
import numpy as np
from sklearn import linear_model


class AVT:
    def __init__(self):
        self.ob_mean = None
        self.ob_std = None
        self.pr_mean = None
        self.pr_std = None

    def fit(self, ob: np.ndarray, pr: np.ndarray) -> None:
        self.ob_mean = np.nanmean(ob)
        self.ob_std = np.nanstd(ob)
        self.pr_mean = np.nanmean(pr)
        self.pr_std = np.nanstd(pr)


    def predict(self, pr0: np.ndarray) -> np.ndarray:
        pr = (pr0 - self.pr_mean) / self.pr_std * self.ob_std + self.ob_mean
        return pr


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


def main(input_path: str, output_path: str) -> None:
    data_list = list()
    label_list = list()
    for i in range(2017, 2023):
        data_list.append(np.load(os.path.join(input_path, 'nwp{:d}.npy'.format(i))))
        label_list.append(np.load(os.path.join(input_path, 'ob{:d}.npy'.format(i))))
        # print(i)

    for i in range(6):
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
        # 方案四
        predict_label = np.zeros_like(nwp_test_label) + np.nan
        for j in range(24):
            for k in range(24):
                model = AVT()
                model.fit(train_label[:, j, k], nwp_train_label[:, j, k])
                predict_label[:, j, k] = model.predict(nwp_test_label[:, j, k])
        # np.save(os.path.join(output_path, f'ob_3_{year}.npy'), test_label)
        np.save(os.path.join(output_path, f'avt_3_{year}.npy'), predict_label)
        # np.save(os.path.join(output_path, f'nwp_3_{year}.npy'), nwp_test_label)
        print(year)


if __name__ == '__main__':
    print('The program "avt.py" is beginning.')
    start = arrow.now()

    main(r'D:\data\wind', r'D:\Project\wind\97')

    end = arrow.now()
    running_time = (end - start).total_seconds()

    print('The program "avt.py" runs out in {:s}.'.format(format_time(running_time)))
