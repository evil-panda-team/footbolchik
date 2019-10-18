# -*- coding: utf-8 -*-

import os 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import cv2
from tqdm import tqdm
from mmdet.datasets import MyDataset

config_file = 'configs/my_cascade_rcnn_x101_32x4d_fpn_1x.py'
checkpoint_file = 'work_dirs/my_plus1_cascade_rcnn_x101_32x4d_fpn_1x/epoch_7.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

vid_filename = '1436406_1_001'
vid_file = "/colddata/datasets/footbolchik/{}.mp4".format(vid_filename)
video = mmcv.VideoReader(vid_file)

thresh = 0.9

out = cv2.VideoWriter('/colddata/datasets/footbolchik/{}_detections.mp4'.format(vid_filename), cv2.VideoWriter_fourcc(*'XVID'), 50, (1920, 1080))

for frame in tqdm(video):
    results = inference_detector(model, frame)
    results = results[:len(MyDataset.CLASSES)]
    o = show_result(frame, results, MyDataset.CLASSES, score_thr=thresh, show = False)
    out.write(o)
out.release()