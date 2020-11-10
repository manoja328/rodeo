#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:09:24 2019

@author: manoj
"""

#%%

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from PIL import Image
from scipy.io import loadmat

#%%

#TODO:  https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py

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

edge_boxes_path_test = '/media/manoj/hdd/VOCdevkit/VOC2007/EdgeboxesProposals/EdgeBoxesVOC2007test.mat'
voc_2007_test = loadmat(edge_boxes_path_test)

edge_boxes_path_trainval = '/media/manoj/hdd/VOCdevkit/VOC2007/EdgeboxesProposals/EdgeBoxesVOC2007trainval.mat'
voc_2007_trainval = loadmat(edge_boxes_path_trainval)

#%%

def get_edbeboxes(data):
    edboxes_voc07 = {}
    scores  = data['boxScores'][0]
    for idx in range(len(scores)):
        image = data['images'][0][idx].item()
        scores  = data['boxScores'][0][idx] # scores are already sorted
        boxes  = data['boxes'][0][idx]
        Nrois = 2000
        #select top 128 highest socring 
        rois  = boxes[:Nrois]
        #put in x0,y0,x1,y1 order
        rois = rois[:,(1,0,3,2)] 
        edboxes_voc07[image] = rois
    return edboxes_voc07
        
        
trainval = get_edbeboxes(voc_2007_trainval)
test = get_edbeboxes(voc_2007_test)
trainval.update(test)   

import pickle

with open("datasets/edboxes_voc07_2000.pkl","wb") as f:
    pickle.dump(trainval,f)
     

#%%
print (voc_2007_test.keys())
idx = 0
image = voc_2007_test['images'][0][idx].item()
scores  = voc_2007_test['boxScores'][0][idx] # scores are already sorted
boxes = voc_2007_test['boxes'][0][idx]
Nrois = 128
#select top 128 highest socring 
rois  = boxes[:Nrois]
#sample 64 randomly out of 128
sample_idx = np.random.choice(Nrois,64)


pil = Image.open("/media/manoj/hdd/VOCdevkit/VOC2007/JPEGImages/{}.jpg".format(image))
imagearr = np.array(pil)
sampled_rois = rois[sample_idx]
sampled_rois = sampled_rois[:,(1,0,3,2)] 
size=(pil.height,pil.width)
print ("H,W",size)
draw_rois(torch.from_numpy(sampled_rois),image=imagearr,size= size)


#%%

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


plt.figure(figsize=(8,10))
plt.imshow(imagearr)
for box in rois[:,(1,0,3,2)]:
   xmin,ymin,xmax,ymax  = box
   x =[xmin,ymin,xmax,ymax]
   rect = retbox(x)
   plt.plot(rect[:,0],rect[:,1],'r',linewidth=2.0)



#%%

SELDIR = '/home/manoj/Desktop/incremental_OD/datasets/voc/VOCdevkit/VOC2007/SelectiveSearchProposals'

import os

rois_selective = np.load(os.path.join(SELDIR,image+'.npy'))
sampled_rois = np.float32(rois_selective[:64])

draw_rois(torch.from_numpy(sampled_rois),image=imagearr,size= size)





#%%



matfile = '/media/manoj/hdd/box_expts/pascal_voc07_candidates_recall/precomputed/edge_boxes_AR/mat/0000/000008.mat'

m = loadmat(matfile)
rois = np.float32(m['boxes'])[:100]
#rois = rois[:,(1,0,3,2)]

pil = Image.open("/media/manoj/hdd/VOCdevkit/VOC2007/JPEGImages/{}.jpg".format('000008'))
imagearr = np.array(pil)
size=(pil.height,pil.width)
print ("H,W",size)
#draw_rois(torch.from_numpy(rois),image=imagearr,size= size)


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


plt.figure(figsize=(8,10))
plt.imshow(imagearr)
for box in rois:
   xmin,ymin,xmax,ymax  = box
   x =[xmin,ymin,xmax,ymax]
   rect = retbox(x)
   plt.plot(rect[:,0],rect[:,1],'r',linewidth=2.0)




#%%
import glob
import os
import pickle
from scipy.io import loadmat
from tqdm import tqdm 
import numpy as np

DIR = '/media/manoj/hdd/box_expts/precomputed-coco/edge_boxes_AR/mat/COCO_train2014'   

files = glob.glob(os.path.join(DIR,"**/*.mat"),recursive=True)
from torchvision.ops.boxes import nms
def filter_small_boxes(boxes, min_size):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = np.where((w >= min_size) & (h >= min_size))[0]
    return keep


def unique_boxes(boxes, scale=1.0):
    """Return indices of unique boxes."""
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)


edboxes_voc07 = {}
for file in tqdm(files):
    image = file.split("/")[-1].replace(".mat","")
    m = loadmat(file)
    Nrois = 2000
      
    boxes = np.float32(m['boxes'])
    scores = np.float32(m['scores'])
    
#    idS = nms(torch.Tensor(boxes),torch.Tensor(scores),0.7)
#    boxes = boxes[idS]
#    
#    keep = unique_boxes(boxes)
#    boxes = boxes[keep, :]
#    keep = filter_small_boxes(boxes, 16)
#    boxes = boxes[keep, :]    
    boxes_final = boxes[:Nrois]    
    edboxes_voc07[image] = boxes_final

with open("datasets/edboxes_coco14_2000_new22.pkl","wb") as f:
    pickle.dump(edboxes_voc07,f)
     

#import pickle 
#files = ["/home/manoj/Desktop/incremental_OD/datasets/edboxes_coco_train2014_2000.pkl",
#"/home/manoj/Desktop/incremental_OD/datasets/edboxes_coco_val2014_2000.pkl"]
#
#d = {}
#for file in files:
#    with open(file,'rb') as f:
#          j = pickle.load(f)
#          for k in j:
#              intk = int(k.split("_")[-1])
#              d[intk] = j[k]
#          
#with open("datasets/edboxes_coco2014trainval_2000.pkl","wb") as f:
#    pickle.dump(d,f)


#%%
    
file = random.choice(files)
image = file.split("/")[-1].replace(".mat","")
m = loadmat(file)
Nrois = 2000
boxes = np.float32(m['boxes'])
scores = np.float32(m['scores'])

pil = Image.open("/media/manoj/hdd/VOCdevkit/VOC2007/JPEGImages/{}.jpg".format(image))
imagearr = np.array(pil)
draw_rois(boxes,imagearr)
draw_rois(boxes_f,imagearr)





