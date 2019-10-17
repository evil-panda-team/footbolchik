# -*- coding: utf-8 -*-

import os 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from mmdet.apis import init_detector, inference_detector
import mmcv
from tqdm import tqdm
import pickle
from mmdet.datasets import MyDataset

config_file = 'configs/my_cascade_rcnn_x101_32x4d_fpn_1x.py'
checkpoint_file = 'work_dirs/my_plus1_cascade_rcnn_x101_32x4d_fpn_1x/epoch_7.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

vid_file = "/colddata/datasets/footbolchik/mp4/013.mp4"
video = mmcv.VideoReader(vid_file)

thresh = 0.9

list_frames = []

for frame in tqdm(video):
    list_frame = []
    results = inference_detector(model, frame)
    results = results[:len(MyDataset.CLASSES)]
    for cl, result in enumerate(results):
        for res in result:
            if res[-1] >= thresh:
                dict_pred = {}
                x_c = (res[2] + res[0])/2
                y_c = (res[3] + res[1])/2
                dict_pred['x'] = int(x_c)
                dict_pred['y'] = int(y_c)
                dict_pred['class'] = MyDataset.CLASSES[cl]
                list_frame.append(dict_pred)
    list_frames.append(list_frame)

print(list_frames)
print(len(list_frames))

with open('/colddata/datasets/footbolchik/predict_013.pkl', 'wb') as f:
    pickle.dump(list_frames, f)