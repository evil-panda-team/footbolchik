# -*- coding: utf-8 -*-

import cv2
from glob import glob
from tqdm import tqdm

# In[]:
folder ='../mmdetection/results/'
size = (1920, 1080)

jpgs = [f for f in glob(folder + '*.jpg', recursive=True)]
jpgs.sort()

out = cv2.VideoWriter('005_detections.mp4', cv2.VideoWriter_fourcc(*'XVID'), 24, size)

for jpg in tqdm(jpgs):
    img = cv2.imread(jpg)
    out.write(img)
out.release()

# In[]:




# In[]:




# In[]:



