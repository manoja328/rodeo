#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:20:40 2020

@author: manoj
"""


import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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

def draw_bboxes(img,ent):
    plt.figure()
    plt.imshow(np.array(img))
    for box in ent['gtbox']:
       rect = retbox(box,'xywh')
       plt.plot(rect[:,0],rect[:,1],'r',linewidth=2.0)
    plt.title(ent['question'])
    plt.axis('off')
    plt.show()


#%%
data = {}
data['train'] = "datasets/vqd/train.json"
data['val'] = "datasets/vqd/val.json"

vqd = {}
for s,file in data.items():
    with open(file) as f:
        vqd[s] = json.load(f)


#%%
root = Path("/media/manoj/hdd/VQA/Images/mscoco")
ent = vqd['train'][903]
path = root / '{}2014'.format(ent['split']) / ent['file_name']
pil  = Image.open(path)
draw_bboxes(pil,ent)

#%%

featspath = Path('/hdd/manoj/cadene_feats/2018-04-27_bottom-up-attention_fixed_36')
import torch 
path = '{}.pth'.format(featspath/ ent['file_name'])
a = torch.load(path)




