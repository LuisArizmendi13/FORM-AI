#!/usr/bin/env python

import numpy as np
import argparse

def distance(pose_g, pose_t, point_idx):
    x_diff = pose_g[point_idx][0] - pose_t[point_idx][0]
    y_diff = pose_g[point_idx][1] - pose_t[point_idx][1]
    z_diff = pose_g[point_idx][2] - pose_t[point_idx][2]
    return np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

def reorient(sequence):
    left_avg = np.mean(sequence[:, 1::2, :], axis=(0, 1))
    right_avg = np.mean(sequence[:, 2::2, :], axis=(0, 1))

    theta_left = np.arctan2(left_avg[1], left_avg[0])
    theta_right = np.arctan2(right_avg[1], right_avg[0])
    angle = -(theta_left + theta_right) / 2

    # Z
    """rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])"""

    # Y
    rotation_matrix = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])

    # X
    """rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])"""

    rotated = np.empty_like(sequence)
    for i in range(sequence.shape[0]):
        for j in range(sequence.shape[1]):
            rotated[i, j, :] = sequence[i, j, :] @ rotation_matrix.T

    return rotated

def dtw(path_g, path_t, justpath=False):
    gold_sequence = np.load(path_g, allow_pickle=True)
    test_sequence = np.load(path_t, allow_pickle=True)

    gold_sequence = reorient(gold_sequence)
    test_sequence = reorient(test_sequence)

    g = len(gold_sequence)
    t = len(test_sequence)

    dists = np.zeros((g, t, 17), dtype=float)

    for i in range(g):
        for j in range(t):
            for k in range(17):
                dists[i, j, k] = distance(gold_sequence[i], test_sequence[j], k)

    def dp_all():  # no index, uses all indexes
        my_cost = np.full((g, t), 9999999, dtype=float)
        my_cost[0, 0] = np.sum(dists[0, 0, :])
        for j in range(1, t): my_cost[0, j] = np.sum(dists[0, j, :]) + my_cost[0, j - 1]
        for i in range(1, g): my_cost[i, 0] = np.sum(dists[i, 0, :]) + my_cost[i - 1, 0]

        for i in range(1, g):
            for j in range(1, t):
                my_cost[i, j] = np.sum(dists[i, j, :]) + min(
                    my_cost[i - 1, j],     # Insertion
                    my_cost[i, j - 1],     # Deletion
                    my_cost[i - 1, j - 1]  # Match
                )

        # find the path
        i, j = g - 1, t - 1
        warp_path = [(i, j)]
        while i > 0 or j > 0:
            if i == 0: j -= 1
            elif j == 0: i -= 1
            else:
                _, i, j = min((my_cost[i - 1, j], i - 1, j),
                              (my_cost[i, j - 1], i, j - 1),
                              (my_cost[i - 1, j - 1], i - 1, j - 1),
                              key = lambda x : x[0])
            warp_path.append((i, j))

        return reversed(warp_path), g > t

    def dp(idx):
        my_cost = np.full((g, t), 9999999, dtype=float)
        my_cost[0, 0] = dists[0, 0, idx]
        for j in range(1, t): my_cost[0, j] = dists[0, j, idx] + my_cost[0, j - 1]
        for i in range(1, g): my_cost[i, 0] = dists[i, 0, idx] + my_cost[i - 1, 0]

        for i in range(1, g):
            for j in range(1, t):
                my_cost[i, j] = dists[i, j, idx] + min(
                    my_cost[i - 1, j],     # Insertion
                    my_cost[i, j - 1],     # Deletion
                    my_cost[i - 1, j - 1]  # Match
                )

        # find the path
        i, j = g - 1, t - 1
        warp_path = [(i, j)]
        while i > 0 or j > 0:
            if i == 0: j -= 1
            elif j == 0: i -= 1
            else:
                _, i, j = min((my_cost[i - 1, j], i - 1, j),
                              (my_cost[i, j - 1], i, j - 1),
                              (my_cost[i - 1, j - 1], i - 1, j - 1),
                              key = lambda x : x[0])
            warp_path.append((i, j))

        # for each T pose, find minimum cost from G
        out = np.full((t), 999999, dtype=float)
        for i, j in reversed(warp_path):
            out[j] = min(out[j], dists[i, j, idx])
        return out

    if justpath: return dp_all()

    sequence_cost = np.zeros((t, 17), dtype=float)
    for k in range(17): sequence_cost[:, k] = dp(k)
    return sequence_cost

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', dest='gold', type=str, default=None)
    parser.add_argument('--test', dest='test', type=str, default=None)
    args = parser.parse_args()
    assert args.gold is not None or args.test is not None, 'missing paths to gold and test poses'
    res = dtw(args.gold, args.test)
    print(res)
