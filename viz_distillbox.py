#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 16:08:15 2019

@author: manoj
"""

import random 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
#%%
VOC_CATS = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
            'tvmonitor']

def retbox(bbox,format='xyxy'):    
    """A utility function to return box coords asvisualizing boxes."""
    if format =='xyxy':
        xmin, ymin, xmax, ymax = bbox
    elif format =='xywh':
        xmin, ymin, w, h = bbox
        xmax = xmin + w -1 
        ymax = ymin + h -1      
    
    box =  np.array([[xmin, xmax, xmax, xmin, xmin],
                    [ymin, ymin, ymax, ymax, ymin]])
    return box.T

def draw_rois(boxes,image=None,size=None):
    plt.figure(figsize=(8,6))
    if image is not None:
        plt.imshow(image)
    for box in boxes:
       xmin,ymin,xmax,ymax  = box
       x =[xmin,ymin,xmax,ymax]
       rect = retbox(x)
       plt.plot(rect[:,0],rect[:,1],'r',linewidth=2.0)


#%%  
iter = 1          
with open("iter{}_models_incr_voc/info.pkl".format(iter),"rb") as f:
    info = pickle.load(f)         
                
keys = list(info.keys())
for key in keys:
    #key = random.choice(keys)
    #keys  = '007664'
    a = info[key]
    boxes = a['proposals']
    image = key
    pil = Image.open("/media/manoj/hdd/VOCdevkit/VOC2007/JPEGImages/{}.jpg".format(image))
    h,w = info[key]['sizes'][0]
    size = (h,w)
    pil = pil.resize((w,h))
    imagearr = np.array(pil)
    draw_rois(boxes,image=imagearr,size= size)
    plt.tight_layout()
    plt.savefig("boxes{}/{}_box.jpg".format(iter,key))
    plt.close()

