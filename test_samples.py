import cv2
import os
import math
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

show_video = False
save_annotations = True

dataset_global = []
test_train_ratio = 0.2
dataset_path = 'samples'
path_to_save = 'annotations/'

CLASSES = ['Pb4 L', 'Pb3 L', 'C2 C', 'C1 C', 'Pb4 R', 'Pb3 R']
point_names = ['Co C', 'C1 C', 'C2 C', 'Cf C', 'Cn C', 'F L', 'N L',
               'Pp L', 'Pb1 L', 'Pb2 L', 'Pb3 L', 'Pb4 L', 'Pb5 L',
               'Pb6 L', 'Gb1 L', 'Gb2 L', 'Gb3 L', 'Gb4 L', 'F R',
               'N R', 'Pp R', 'Pb1 R', 'Pb2 R', 'Pb3 R', 'Pb4 R',
               'Pb5 R', 'Pb6 R', 'Gb1 R', 'Gb2 R', 'Gb3 R', 'Gb4 R']

points_pairs_to_check = [['Pb4 L', 'Pb3 L'],
                         ['C2 C', 'C1 C'],
                         ['Pb4 R', 'Pb3 R']]
points_pairs_to_check_idxs = []
for points in points_pairs_to_check:
    pt1_idx = point_names.index(points[0])
    pt2_idx = point_names.index(points[1])
    points_pairs_to_check_idxs.append([pt1_idx, pt2_idx])

if show_video:
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    font = cv2.FONT_HERSHEY_SIMPLEX

w = 1920
h = 1080
no_point_counter = 0
for i in range(10):
    sample_number = i+1
    if sample_number == 10:
        video_file_name = 'video_010.mp4'
        log_file_name = 'log_010.csv'
        annotations_folder = '010'
    elif sample_number == 5:
        continue
    else:
        video_file_name = 'video_00{}.mp4'.format(sample_number)
        log_file_name = 'log_00{}.csv'.format(sample_number)
        annotations_folder = '00{}'.format(sample_number)
    if show_video:
        cap = cv2.VideoCapture(os.path.join(dataset_path, video_file_name))
        if (cap.isOpened() == False):
            print("Error opening video stream or file")

    with open(os.path.join(dataset_path, log_file_name)) as f:
        points_list = [line.split(',') for line in f]

    reference_sizes = []

    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    bbox_w = 25
    bbox_h = 25

    scale = 1

    for frame_number in range(len(points_list)):
        if sample_number == 1:
            if frame_number > 4320:
                break
        if frame_number == 0:
            continue
        if show_video:
            ret, frame = cap.read()
        coords = points_list[frame_number][1:]
        coord_points = [[coords[2*i], coords[2*i+1]]
                        for i in range(len(coords)//2)]
        if frame_number == 1:
            for points_idxs in points_pairs_to_check_idxs:
                pt1 = coord_points[points_idxs[0]]
                pt2 = coord_points[points_idxs[1]]
                distance = math.sqrt(
                    sum([(int(a) - int(b)) ** 2 for a, b in zip(pt1, pt2)]))
                reference_sizes.append(distance)
            print(reference_sizes)
        elif frame_number < 150 and frame_number > 1:
            continue
        else:
            for idx, points_idxs in enumerate(points_pairs_to_check_idxs):
                if coord_points[points_idxs[0]][0] != '' and coord_points[points_idxs[1]][0] != '':
                    pt1 = coord_points[points_idxs[0]]
                    pt2 = coord_points[points_idxs[1]]
                    distance = math.sqrt(
                        sum([(int(a) - int(b)) ** 2 for a, b in zip(pt1, pt2)]))
                    scale = distance / reference_sizes[idx]

        bboxes = []
        labels = []
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0, ))
        no_point = True
        # draw field points
        for idx, point in enumerate(coord_points):
            if point[0] != '':
                x, y = int(point[0]), int(point[1])
                x1 = int(x - scale*bbox_w/2)
                x2 = int(x + scale*bbox_w/2)
                y1 = int(y - scale*bbox_h/2)
                y2 = int(y + scale*bbox_h/2)
                if point_names[idx] in CLASSES:
                    bboxes.append([x1, y1, x2, y2])
                    labels.append(CLASSES.index(point_names[idx])+1)
                    no_point = False
                if show_video:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, point_names[idx],
                                (int(point[0]), int(point[1])),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 255), 2)
        if no_point:
            no_point_counter += 1
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2)
            labels = np.array(labels)

        annotation = {
            'filename': '{}/{}.png'.format(annotations_folder, str(frame_number-1)),
                        'width': w,
                        'height': h,
                        'ann': {
                            'bboxes': bboxes.astype(np.float32),
                            'labels': labels.astype(np.int64),
                            'bboxes_ignore': bboxes_ignore.astype(np.float32),
                            'labels_ignore': labels_ignore.astype(np.int64)
            }
        }
        dataset_global.append(annotation)

        if show_video:
            if ret == True:
                cv2.imshow("output", frame)
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
    if show_video:
        cap.release()
        cv2.destroyAllWindows()

print(no_point_counter)
if save_annotations:
    with open('{}data_global.pkl'.format(path_to_save), 'wb') as f:
        pickle.dump(dataset_global, f)

    dataset_train, dataset_val = train_test_split(
        dataset_global, test_size=test_train_ratio, random_state=42, shuffle=True)

    with open('{}data_train.pkl'.format(path_to_save), 'wb') as f:
        pickle.dump(dataset_train, f)

    with open('{}data_val.pkl'.format(path_to_save), 'wb') as f:
        pickle.dump(dataset_val, f)
