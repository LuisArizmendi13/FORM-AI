#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

def draw_skeleton(pose_3d, ax=None):
    if ax is not None: ax.clear()

    skeleton_edges = [
        (0, 1), (1, 2), (2, 3),  # Right leg
        (0, 4), (4, 5), (5, 6),  # Left leg
        (0, 7), (7, 8), (8, 9), (9, 10),  # Spine to head
        (8, 11), (11, 12), (12, 13),  # Left arm
        (8, 14), (14, 15), (15, 16)   # Right arm
    ]

    # Plot keypoints as red dots
    ax.scatter(pose_3d[:, 0], pose_3d[:, 2], -pose_3d[:, 1], c='r', marker='o')

    # Plot skeleton connections as blue lines
    for edge in skeleton_edges:
        point1, point2 = edge
        ax.plot([pose_3d[point1, 0], pose_3d[point2, 0]],
                [pose_3d[point1, 2], pose_3d[point2, 2]],
                [-pose_3d[point1, 1], -pose_3d[point2, 1]], c='b')

    # Set labels and adjust view
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Human Pose in Human3.6M Format')
    ax.view_init(elev=0, azim=90)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

if __name__ == '__main__':
    points_path = sys.argv[1]
    poses = np.load(points_path, allow_pickle=True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    def update(frame):
        draw_skeleton(pose_3d=poses[frame], ax=ax)

    anim = FuncAnimation(fig=fig, func=update, frames=len(poses), interval=500)
    anim.save('skeleton.gif', writer=PillowWriter(fps=10))
