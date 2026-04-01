#!user/bin.python3

"""
Founded in 2024-07-27
Modified in 2024-07-30
@author: yinlb
"""
import os
import sys
import typing

import arrow
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt


def floyd(graph: np.ndarray) -> [np.ndarray, np.ndarray]:
    n = graph.shape[0]
    distance = np.copy(graph)
    node_ind = np.zeros_like(graph, dtype=np.int_) - 1
    for i in range(n):
        for j in range(n):
            if graph[i, j] < np.inf:
                node_ind[i, j] = j
            else:
                node_ind[i, j] = -1
    for k in range(n):
        for i in range(n):
            if distance[i, k] < np.inf:
                for j in range(n):
                    if distance[i, j] > distance[i, k] + distance[k, j]:
                        distance[i, j] = distance[i, k] + distance[k, j]
                        node_ind[i, j] = k
    return [distance, node_ind]


def n_edge(node_ind: np.ndarray) -> np.ndarray:
    n = node_ind.shape[0]
    num_edge = np.zeros_like(node_ind, dtype=np.int_)
    for i in range(n):
        for j in range(n):
            line_list = list()
            line_list.append([i, j])
            while len(line_list) > 0:
                x, y = line_list.pop()
                k = node_ind[x, y]
                if k < 0:
                    continue
                elif k == y:
                    num_edge[x, y] += 1
                else:
                    line_list.append([x, k])
                    line_list.append([k, y])
    return num_edge


def get_community(node_ind: np.ndarray) -> np.ndarray:
    n = node_ind.shape[0]
    community = np.zeros(n, dtype=np.int_) - 1
    index = community > 0
    next_id = 0
    for i in range(n):
        if index[i]:
            continue
        line_list = [i]
        while len(line_list) > 0:
            j = line_list.pop()
            if index[j]:
                continue
            community[j] = next_id
            index[j] = True
            for k in range(n):
                if node_ind[j, k] != -1:
                    line_list.append(node_ind[j, k])
        next_id += 1
        if np.sum(~index) == 0:
            break
    return community


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


def main() -> None:
    rmse_cross = np.load(r'D:\Project\wind\97\rmse_cross.npy')
    graph = np.copy(rmse_cross)
    for i in range(97):
        for j in range(97):
            temp = rmse_cross[i, j] * rmse_cross[j, i] - rmse_cross[i, i] * rmse_cross[j, j]
            graph[i, j] = temp
    # for i in range(97):
    #     for j in range(97):
    #         temp = rmse_cross[i, j] + rmse_cross[j, i]
    #         graph[i, j] = temp
    print(np.min(graph))
    print(np.max(graph))
    graph[graph < 0] = 0
    graph[graph > 0.5] = np.inf
    mask = np.triu(np.ones_like(graph, dtype=bool))
    np.fill_diagonal(graph, np.inf)
    a = 0
    while True:
        # print(np.sum(graph < np.inf) / 2)
        distance, node_ind = floyd(graph)
        num_edge = n_edge(node_ind)
        # num_edge[mask] = -1
        max_num_edge = np.max(num_edge)
        # print(max_num_edge)
        index = num_edge == max_num_edge
        graph[index] = np.inf
        if a != np.sum(distance == np.inf):
            community = get_community(node_ind)
            a = np.sum(distance == np.inf)
            np.save(rf'D:\Project\wind\97\gn\graph-{a}.npy', graph)
            np.save(rf'D:\Project\wind\97\gn\distance-{a}.npy', distance)
            np.save(rf'D:\Project\wind\97\gn\node_ind-{a}.npy', node_ind)
            n_communities = np.max(community) + 1
            with open(rf'D:\Project\wind\97\gn\community-{a}.txt', mode='w') as fid:
                for i in range(n_communities):
                    txt = list()
                    for j in range(97):
                        if community[j] == i:
                            txt.append(str(j))
                    txt = ','.join(txt) + '\n'
                    fid.write(txt)
        if np.sum(distance < np.inf) == 0:
            break
    # # np.fill_diagonal(mask, False)
    # sb.heatmap(distance, mask=mask, cmap='Reds', linewidths=0.3)
    # plt.tick_params(axis='x', which='both', bottom=False, top=False)
    # plt.tick_params(axis='y', which='both', left=False, right=False)
    # plt.savefig(r'D:\Project\wind\97\rmse_cross12133.png', bbox_inches='tight', dpi=300)
    # plt.close()


if __name__ == '__main__':
    print('The program "gn.py" is beginning.')
    start = arrow.now()

    main()

    end = arrow.now()
    running_time = (end - start).total_seconds()

    print('The program "gn.py" runs out in {:s}.'.format(format_time(running_time)))
