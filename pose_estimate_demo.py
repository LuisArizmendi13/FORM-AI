#!/usr/bin/env python3

import os
import subprocess
import shutil

# see: https://github.com/facebookresearch/VideoPose3D/blob/main/INFERENCE.md
# takes input videos (in video_in dir) and produces 3d joint positions
# in Human3.6M format (DIR HERE)

IN_VIDEO_NAME = 'walking.mp4'  # for debugging, temp fix

def main():
    PROJECT_DIR = os.getcwd()
    if '/' not in PROJECT_DIR or PROJECT_DIR[PROJECT_DIR.rfind('/')+1:] != 'FORM-AI':
        assert False, 'not running from project directory'

    print('FORM-AI pose_estimate_demo: run detectron2 to create 2d input keypoints in COCO format')
    os.chdir('./videopose3d/inference')
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
    try: shutil.rmtree(PROJECT_DIR + '/dataset_out')
    except: pass  # only delete if exists
    shutil.copytree(os.getcwd(), PROJECT_DIR + '/dataset_out')

    print('FORM-AI pose_estimate_demo: render output and export coordinates')
    os.chdir('..')  # to PROJECT_DIR/videopose3d
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

if __name__ == '__main__':
    main()
