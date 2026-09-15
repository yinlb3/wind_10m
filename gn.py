#!user/bin.python3
# -*- coding: utf-8 -*-

"""
Founded in 2024-07-27
Modified in 2026-09-15
@author: yinlb
"""

import os
import pathlib
import typing

import arrow
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from src import utils


def _floyd(graph: np.ndarray) -> typing.List[np.ndarray]:
    """Compute all-pairs shortest paths using Floyd-Warshall algorithm.

    Args:
        graph: Adjacency matrix with edge weights; np.inf means no edge.

    Returns:
        A list containing the shortest-distance matrix and the intermediate
        node index matrix.
    """
    n_nodes = graph.shape[0]
    distance = np.copy(graph)
    node_ind = np.zeros_like(graph, dtype=np.int_) - 1

    # 1. Initialize the intermediate node index matrix.
    for i in range(n_nodes):
        for j in range(n_nodes):
            if graph[i, j] < np.inf:
                node_ind[i, j] = j
            else:
                node_ind[i, j] = -1

    # 2. Iteratively relax the shortest paths.
    for k in range(n_nodes):
        for i in range(n_nodes):
            if distance[i, k] < np.inf:
                for j in range(n_nodes):
                    if distance[i, j] > distance[i, k] + distance[k, j]:
                        distance[i, j] = distance[i, k] + distance[k, j]
                        node_ind[i, j] = k

    return [distance, node_ind]


def _n_edge(node_ind: np.ndarray) -> np.ndarray:
    """Count the number of edges on shortest paths between all node pairs.

    Args:
        node_ind: Intermediate node index matrix from `_floyd`.

    Returns:
        Matrix with edge counts for each node pair.
    """
    n_nodes = node_ind.shape[0]
    num_edge = np.zeros_like(node_ind, dtype=np.int_)
    for i in range(n_nodes):
        for j in range(n_nodes):
            line_list = list()
            line_list.append([i, j])
            while len(line_list) > 0:
                node_a, node_b = line_list.pop()
                k = node_ind[node_a, node_b]
                if k < 0:
                    continue
                elif k == node_b:
                    num_edge[node_a, node_b] += 1
                else:
                    line_list.append([node_a, k])
                    line_list.append([k, node_b])
    return num_edge


def _get_community(node_ind: np.ndarray) -> np.ndarray:
    """Identify connected communities from the path index matrix.

    Args:
        node_ind: Intermediate node index matrix from `_floyd`.

    Returns:
        Array assigning each node to a community id.
    """
    n_nodes = node_ind.shape[0]
    community = np.zeros(n_nodes, dtype=np.int_) - 1
    visited = np.zeros(n_nodes, dtype=bool)
    next_id = 0

    for i in range(n_nodes):
        if visited[i]:
            continue
        line_list = [i]
        while len(line_list) > 0:
            j = line_list.pop()
            if visited[j]:
                continue
            community[j] = next_id
            visited[j] = True
            for k in range(n_nodes):
                if node_ind[j, k] != -1:
                    line_list.append(node_ind[j, k])
        next_id += 1
        if np.sum(~visited) == 0:
            break

    return community


def main() -> None:
    """Run graph-based community detection on the station RMSE matrix."""
    # 1. Load configuration and input data.
    cfg = utils.load_config()
    project_wind = pathlib.Path(cfg['paths']['project_wind'])
    station_num = cfg['constants']['station_num']
    input_path = project_wind / '97' / 'rmse_cross.npy'
    rmse_cross = np.load(input_path)

    # 2. Build the affinity graph from the cross-RMSE matrix.
    graph = np.copy(rmse_cross)
    for i in range(station_num):
        for j in range(station_num):
            link_score = (
                rmse_cross[i, j] * rmse_cross[j, i]
                - rmse_cross[i, i] * rmse_cross[j, j]
            )
            graph[i, j] = link_score

    # 3. Threshold the graph and detect communities iteratively.
    print(np.min(graph))
    print(np.max(graph))
    graph[graph < 0] = 0
    graph[graph > 0.5] = np.inf
    mask = np.triu(np.ones_like(graph, dtype=bool))
    np.fill_diagonal(graph, np.inf)

    prev_inf_count = 0
    while True:
        distance, node_ind = _floyd(graph)
        num_edge = _n_edge(node_ind)
        max_num_edge = np.max(num_edge)
        index = num_edge == max_num_edge
        graph[index] = np.inf
        cur_inf_count = np.sum(distance == np.inf)
        if prev_inf_count != cur_inf_count:
            community = _get_community(node_ind)
            prev_inf_count = cur_inf_count
            out_dir = project_wind / '97' / 'gn'
            os.makedirs(out_dir, exist_ok=True)
            np.save(out_dir / f'graph-{cur_inf_count}.npy', graph)
            np.save(out_dir / f'distance-{cur_inf_count}.npy', distance)
            np.save(out_dir / f'node_ind-{cur_inf_count}.npy', node_ind)
            n_communities = np.max(community) + 1
            out_file = out_dir / f'community-{cur_inf_count}.txt'
            with open(out_file, mode='w') as fid:
                for i in range(n_communities):
                    txt = list()
                    for j in range(station_num):
                        if community[j] == i:
                            txt.append(str(j))
                    txt = ','.join(txt) + '\n'
                    fid.write(txt)
        if np.sum(distance < np.inf) == 0:
            break

    # Optional heatmap output (currently disabled).
    # sb.heatmap(distance, mask=mask, cmap='Reds', linewidths=0.3)
    # plt.tick_params(axis='x', which='both', bottom=False, top=False)
    # plt.tick_params(axis='y', which='both', left=False, right=False)
    # plt.savefig(
    #     project_wind / '97' / 'rmse_cross12133.png',
    #     bbox_inches='tight',
    #     dpi=300,
    # )
    # plt.close()


if __name__ == '__main__':
    print('Program gn.py started')
    total_start = arrow.now()

    main()

    total_elapsed = (arrow.now() - total_start).total_seconds()
    print(f'Program gn.py finished, total time '
          f'{utils.format_time(total_elapsed)}')
