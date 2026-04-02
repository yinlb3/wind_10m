#!user/bin.python3

"""
Founded in 2024-07-27
Modified in 2024-07-27
@author: yinlb
"""
import os
import sys

import arrow
import numpy as np


THRES = (0.3, 1.6, 3.4, 5.5, 8.0)
E = 1e-7


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


def ots(train_nwp: np.ndarray, train_ob: np.ndarray, test_nwp: np.ndarray) -> np.ndarray:
    thres0 = list(THRES).copy()
    thres0.reverse()
    t = 11
    thres = list()
    t0max = 100
    tmax = 100
    for t0 in thres0:
        if np.sum(train_ob >= t0) == 0:
            thres.append(t0)
            tmax = t0
            t0max = t0
            continue
        ts_list = list()
        t_list = list()
        while abs(t) > E:
            na = np.sum(((train_nwp >= t) & (train_nwp < tmax)) & ((train_ob >= t0) & (train_nwp < t0max)))
            nb = np.sum(((train_nwp >= t) & (train_nwp < tmax)) & ((train_ob < t0) | (train_nwp >= t0max)))
            nc = np.sum(((train_nwp < t) | (train_nwp >= tmax)) & ((train_ob >= t0) & (train_nwp < t0max)))
            ts_list.append(na / (na + nb + nc + E))
            t_list.append(t)
            t -= 0.1
        t = t_list[np.argmax(ts_list)]
        thres.append(t)
        tmax = t
        t0max = t0
    thres.reverse()
    thres0.reverse()
    pr = np.zeros_like(test_nwp, dtype=np.float32) + np.nan
    index = test_nwp < thres[0]
    pr[index] = test_nwp[index] / thres[0] * thres0[0]
    for i in range(len(thres) - 1):
        index = (test_nwp >= thres[i]) & (test_nwp < thres[i + 1])
        pr[index] = (test_nwp[index] - thres[i]) / (thres[i + 1] - thres[i]) * (thres0[i + 1] - thres0[i]) + thres0[i]
    index = test_nwp >= thres[-1]
    pr[index] = test_nwp[index] / thres[-1] * thres0[-1]
    # print(thres)

    return pr


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
        # 方案一
        predict_label = ots(nwp_train_label, train_label, nwp_test_label)
        np.save(os.path.join(output_path, f'ob_1_{year}.npy'), test_label)
        np.save(os.path.join(output_path, f'ots_1_{year}.npy'), predict_label)
        np.save(os.path.join(output_path, f'nwp_1_{year}.npy'), nwp_test_label)
        # 方案二
        predict_label = np.zeros_like(nwp_test_label) + np.nan
        for j in range(24):
            predict_label[:, j, :] = ots(nwp_train_label[:, j, :], train_label[:, j, :], nwp_test_label[:, j, :])
        np.save(os.path.join(output_path, f'ob_2_{year}.npy'), test_label)
        np.save(os.path.join(output_path, f'ots_2_{year}.npy'), predict_label)
        np.save(os.path.join(output_path, f'nwp_2_{year}.npy'), nwp_test_label)
        # 方案三
        predict_label = np.zeros_like(nwp_test_label) + np.nan
        for j in range(97):
            predict_label[:, :, j] = ots(nwp_train_label[:, :, j], train_label[:, :, j], nwp_test_label[:, :, j])
        np.save(os.path.join(output_path, f'ob_3_{year}.npy'), test_label)
        np.save(os.path.join(output_path, f'ots_3_{year}.npy'), predict_label)
        np.save(os.path.join(output_path, f'nwp_3_{year}.npy'), nwp_test_label)
        # 方案四
        predict_label = np.zeros_like(nwp_test_label) + np.nan
        for j in range(24):
            for k in range(24):
                predict_label[:, j, k] = ots(nwp_train_label[:, j, k], train_label[:, j, k], nwp_test_label[:, j, k])
        np.save(os.path.join(output_path, f'ob_4_{year}.npy'), test_label)
        np.save(os.path.join(output_path, f'ots_4_{year}.npy'), predict_label)
        np.save(os.path.join(output_path, f'nwp_4_{year}.npy'), nwp_test_label)
        print(year)


if __name__ == '__main__':
    print('The program "ots.py" is beginning.')
    start = arrow.now()

    main(r'D:\data\wind', r'D:\Project\wind\97')

    end = arrow.now()
    running_time = (end - start).total_seconds()

    print('The program "ots.py" runs out in {:s}.'.format(format_time(running_time)))
