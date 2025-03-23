#!/usr/bin/env python3

"""
This demo runs inference of the VideoPose3D model from meta
Generates a video comparison of the input and a pose
Takes an mp4 input, placed in the video_in folder
"""

import os
import subprocess
import shutil

# see: https://github.com/facebookresearch/VideoPose3D/blob/main/INFERENCE.md
# takes input videos (in video_in dir) and produces 3d joint positions
# in Human3.6M format (DIR HERE)

# IN_VIDEO_NAME = 'gold_deadlift.mp4'  # for debugging, temp fix
VIDEO = False  # if false, export points instead

def main():
    PROJECT_DIR = os.getcwd()
    if '/' not in PROJECT_DIR or PROJECT_DIR[PROJECT_DIR.rfind('/')+1:] != 'FORM-AI':
        assert False, 'not running from project directory'

    files = [f for f in os.listdir(PROJECT_DIR + '/video_in') if os.path.isfile(os.path.join(PROJECT_DIR + '/video_in', f))]
    assert len(files) == 1, 'make sure exactly one video is in video_in directory'
    IN_VIDEO_NAME = files[0]

    print('FORM-AI pose_estimate_demo: run detectron2 to create 2d input keypoints in COCO format')
    os.chdir('./videopose3d/inference')
    try: shutil.rmtree(PROJECT_DIR + '/keypoints')
    except: pass  # only delete if exists
    os.mkdir(PROJECT_DIR + '/keypoints')
    subprocess.run(['python', 'infer_video_d2.py',
                    '--cfg', 'COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml',
                    '--output-dir', '../../detectron_out',
                    '--image-ext', 'mp4',
                    '../../video_in'])
    # NOTE: if the above does not run, change infer_video_d2.py according
    #     to https://github.com/facebookresearch/VideoPose3D/issues/243

    print('FORM-AI pose_estimate_demo: prepare dataset for videopose3d')
    os.chdir('../data')
    subprocess.run(['python', 'prepare_data_2d_custom.py',
                    '-i', '../../detectron_out',
                    '-o', 'dataset_out_detectron_out'])

    print('FORM-AI pose_estimate_demo: render output and export coordinates')
    os.chdir(PROJECT_DIR + '/videopose3d')  # to PROJECT_DIR/videopose3d
    if VIDEO:
        subprocess.run(['python', 'run.py',
                        # '-d', '../dataset_out',
                        '-d', 'custom_dataset_out',
                        '-k', 'detectron_out',
                        '-arc', '3,3,3,3,3',
                        '-c', 'checkpoint',
                        '--evaluate', 'pretrained_h36m_detectron_coco.bin',
                        '--render',
                        '--viz-subject', f'{IN_VIDEO_NAME}',  # TODO: make this better
                        '--viz-action', 'custom',
                        '--viz-camera', '0',
                        '--viz-video', f'../video_in/{IN_VIDEO_NAME}',  # TODO: make this better
                        '--viz-output', f'../videopose3d_out/{IN_VIDEO_NAME}',  # TODO: make this better
                        '--viz-size', '6'])
    else:
        subprocess.run(['python', 'run.py',
                        '-d', 'custom_dataset_out',
                        '-k', 'detectron_out',
                        '-arc', '3,3,3,3,3',
                        '-c', 'checkpoint',
                        '--evaluate', 'pretrained_h36m_detectron_coco.bin',
                        '--render',
                        '--viz-subject', f'{IN_VIDEO_NAME}',  # TODO: make this better
                        '--viz-action', 'custom',
                        '--viz-camera', '0',
                        '--viz-export', f'../videopose3d_out/{IN_VIDEO_NAME}',  # TODO: make this better
                        '--viz-size', '6'])

if __name__ == '__main__':
    main()
