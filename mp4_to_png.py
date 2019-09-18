# -*- coding: utf-8 -*-

import cv2
from glob import glob
import os
from tqdm import tqdm

# In[]:
folder ='../../../colddata/datasets/footbolchik/mp4/'
mp4s = [f for f in glob(folder + '*.mp4', recursive=True)]

for i, mp4 in tqdm(enumerate(mp4s)):
    vidcap = cv2.VideoCapture(mp4)
    success,image = vidcap.read()
    count = 0
    
    newfolder = '../../../colddata/datasets/footbolchik/png/00{}'.format(i+1)
    
    if not os.path.exists(newfolder):
        os.makedirs(newfolder)
    
    while success:
      cv2.imwrite("{}/{}.png".format(newfolder,count), image)     # save frame as JPEG file      
      success,image = vidcap.read()
#      print('Read a new frame: ', success)
      count += 1

# In[]:




# In[]:




# In[]:




# In[]:



