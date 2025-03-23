#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from dtw_score import dtw

def draw_skeleton(pose_strong, pose_weak, ax=None):
    if ax is not None: ax.clear()

    skeleton_edges = [
        (0, 1), (1, 2), (2, 3),  # Right leg
        (0, 4), (4, 5), (5, 6),  # Left leg
        (0, 7), (7, 8), (8, 9), (9, 10),  # Spine to head
        (8, 11), (11, 12), (12, 13),  # Left arm
        (8, 14), (14, 15), (15, 16)   # Right arm
    ]

    ax.scatter(pose_strong[:, 0], pose_strong[:, 2], -pose_strong[:, 1], c='b', marker='o')
    for edge in skeleton_edges:
        point1, point2 = edge
        ax.plot([pose_strong[point1, 0], pose_strong[point2, 0]],
                [pose_strong[point1, 2], pose_strong[point2, 2]],
                [-pose_strong[point1, 1], -pose_strong[point2, 1]], c='b')

    ax.scatter(pose_weak[:, 0], pose_weak[:, 2], -pose_weak[:, 1], c='r', marker='o')
    for edge in skeleton_edges:
        point1, point2 = edge
        ax.plot([pose_weak[point1, 0], pose_weak[point2, 0]],
                [pose_weak[point1, 2], pose_weak[point2, 2]],
                [-pose_weak[point1, 1], -pose_weak[point2, 1]], c='r', linestyle=':')

    # Set labels and adjust view
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Human Pose in Human3.6M Format')
    ax.view_init(elev=0, azim=90)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

def interpolate_my_skeleton(pose_list, target_length):
    interpolated = []
    for i in range(target_length):
        t = i / (target_length - 1)
        lower_idx = int(t * (len(pose_list) - 1))
        upper_idx = min(lower_idx + 1, len(pose_list) - 1)
        alpha = t * (len(pose_list) - 1) - lower_idx
        interpolated.append((1 - alpha) * pose_list[lower_idx] + alpha * pose_list[upper_idx])
    return interpolated

if __name__ == '__main__':
    points_path_gold = sys.argv[1]
    poses_gold = np.load(points_path_gold, allow_pickle=True)
    points_path_test = sys.argv[2]
    poses_test = np.load(points_path_test, allow_pickle=True)
    # target_length = max(len(poses_gold), len(poses_test))
    # poses_gold = interpolate_my_skeleton(poses_gold, target_length)
    # poses_test = interpolate_my_skeleton(poses_test, target_length)

    matching_path, gold_longer = dtw(points_path_gold, points_path_test, justpath=True)
    gold_and_test = [[], []]
    gold_and_test[1 - gold_longer] = (poses_gold, poses_test)[1 - gold_longer]

    last_off_coord = None
    for match in matching_path:
        if last_off_coord != match[1-gold_longer]: gold_and_test[gold_longer].append((poses_gold, poses_test)[gold_longer][match[gold_longer]])
        last_off_coord = match[1-gold_longer]
    poses_gold, poses_test = gold_and_test
    assert len(poses_gold) == len(poses_test)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    def update(frame):
        draw_skeleton(pose_strong=poses_gold[frame], pose_weak=poses_test[frame], ax=ax)

    anim = FuncAnimation(fig=fig, func=update, frames=len(poses_gold), interval=500)
    anim.save('skeletons.gif', writer=PillowWriter(fps=10))
