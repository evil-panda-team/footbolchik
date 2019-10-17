import cv2
import os
import math
import numpy as np
from sklearn.model_selection import train_test_split
import pickle


def l2_dist(pt1, pt2):
    return math.sqrt(
        sum([(int(a) - int(b)) ** 2 for a, b in zip(pt1, pt2)]))


l2_threshold = 25
dataset_path = 'samples'
path_to_save = 'annotations/'

# For 10 keypoints
# CLASSES = ['F L', 'Pb4 L', 'Pb3 L', 'Cf C', 'C2 C',
#            'C1 C', 'Cn C', 'Pb4 R', 'Pb3 R', 'F R']

# For 6 keypoints
CLASSES = ['Pb4 L', 'Pb3 L', 'C2 C', 'C1 C', 'Pb4 R', 'Pb3 R']

point_names = ['Co C', 'C1 C', 'C2 C', 'Cf C', 'Cn C', 'F L', 'N L',
               'Pp L', 'Pb1 L', 'Pb2 L', 'Pb3 L', 'Pb4 L', 'Pb5 L',
               'Pb6 L', 'Gb1 L', 'Gb2 L', 'Gb3 L', 'Gb4 L', 'F R',
               'N R', 'Pp R', 'Pb1 R', 'Pb2 R', 'Pb3 R', 'Pb4 R',
               'Pb5 R', 'Pb6 R', 'Gb1 R', 'Gb2 R', 'Gb3 R', 'Gb4 R']

log_file_name = 'samples/log_013.csv'
predictions_file_name = 'predictions/predict_013.pkl'

with open(predictions_file_name, 'rb') as f:
    predicted_points = pickle.load(f)

with open(log_file_name) as f:
    points_list = [line.split(',') for line in f]


print(len(points_list))
print(len(predicted_points))
tps = 0
fps = 0
fns = 0
for frame_number in range(len(predicted_points)):
    if frame_number < 240:
        continue
    coords = points_list[frame_number][1:]
    coord_points = [[coords[2*i], coords[2*i+1]]
                    for i in range(len(coords)//2)]

    gt_points = []
    for idx, point in enumerate(coord_points):
        if point[0] != '':
            x, y = int(point[0]), int(point[1])
            if point_names[idx] in CLASSES:
                gt_points.append(
                    {'x': x, 'y': y, 'class': point_names[idx]})

    for pt in predicted_points[frame_number]:
        min_dist = 1000000
        min_idx = 0
        min_class = ''
        for idx, gt_pt in enumerate(gt_points):
            dist = l2_dist([pt['x'], pt['y']], [gt_pt['x'], gt_pt['y']])
            if dist < min_dist:
                min_dist = dist
                min_idx = idx
                min_class = gt_pt['class']
        if min_dist <= l2_threshold and min_class == pt['class']:
            tps += 1
            gt_points.pop(min_idx)
        else:
            fps += 1
    fns += len(gt_points)

precision = tps / (tps+fps)
recall = tps / (tps+fns)
f1 = 2 * precision * recall / (precision + recall)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('f1: {}'.format(f1))
