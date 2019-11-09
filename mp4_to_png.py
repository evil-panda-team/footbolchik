# -*- coding: utf-8 -*-

import cv2
from glob import glob
import os
from tqdm import tqdm

# In[]:
#folder ='../../../colddata/datasets/footbolchik/mp4/'
folder ='../../datasets/footbolchik/mp4/'
mp4s = [f for f in glob(folder + '*.mp4', recursive=True)]
mp4s.sort()

skip = 2
start_sec = 10

for mp4 in tqdm(mp4s):
    vidcap = cv2.VideoCapture(mp4)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success,image = vidcap.read()
    count = 0
    
    newfolder = '../../datasets/footbolchik/png/{}'.format(mp4.split('.mp4')[0][-3:])
    
    if not os.path.exists(newfolder):
        os.makedirs(newfolder)
    
    while success:
        if not count%skip and count >= start_sec*fps:
            cv2.imwrite("{}/{}.png".format(newfolder,count), image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1

# In[]:




# In[]:




# In[]:




# In[]:



