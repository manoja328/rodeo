#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 16:28:55 2019

@author: manoj
"""
from PIL import Image
import torch
import os
import numpy as np
from voc_loader import imgtransform

COCO_VOC_CATS = ['__background__', 'airplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table',
                 'dog', 'horse', 'motorcycle', 'person', 'potted plant',
                 'sheep', 'couch', 'train', 'tv']

COCO_NONVOC_CATS = ['apple', 'backpack', 'banana', 'baseball bat',
                    'baseball glove', 'bear', 'bed', 'bench', 'book', 'bowl',
                    'broccoli', 'cake', 'carrot', 'cell phone', 'clock', 'cup',
                    'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee',
                    'giraffe', 'hair drier', 'handbag', 'hot dog', 'keyboard',
                    'kite', 'knife', 'laptop', 'microwave', 'mouse', 'orange',
                    'oven', 'parking meter', 'pizza', 'refrigerator', 'remote',
                    'sandwich', 'scissors', 'sink', 'skateboard', 'skis',
                    'snowboard', 'spoon', 'sports ball', 'stop sign',
                    'suitcase', 'surfboard', 'teddy bear', 'tennis racket',
                    'tie', 'toaster', 'toilet', 'toothbrush', 'traffic light',
                    'truck', 'umbrella', 'vase', 'wine glass', 'zebra']

COCO_CATS = COCO_VOC_CATS+COCO_NONVOC_CATS

coco_ids = {'airplane': 5, 'apple': 53, 'backpack': 27, 'banana': 52,
            'baseball bat': 39, 'baseball glove': 40, 'bear': 23, 'bed': 65,
            'bench': 15, 'bicycle': 2, 'bird': 16, 'boat': 9, 'book': 84,
            'bottle': 44, 'bowl': 51, 'broccoli': 56, 'bus': 6, 'cake': 61,
            'car': 3, 'carrot': 57, 'cat': 17, 'cell phone': 77, 'chair': 62,
            'clock': 85, 'couch': 63, 'cow': 21, 'cup': 47, 'dining table':
            67, 'dog': 18, 'donut': 60, 'elephant': 22, 'fire hydrant': 11,
            'fork': 48, 'frisbee': 34, 'giraffe': 25, 'hair drier': 89,
            'handbag': 31, 'horse': 19, 'hot dog': 58, 'keyboard': 76, 'kite':
            38, 'knife': 49, 'laptop': 73, 'microwave': 78, 'motorcycle': 4,
            'mouse': 74, 'orange': 55, 'oven': 79, 'parking meter': 14,
            'person': 1, 'pizza': 59, 'potted plant': 64, 'refrigerator': 82,
            'remote': 75, 'sandwich': 54, 'scissors': 87, 'sheep': 20, 'sink':
            81, 'skateboard': 41, 'skis': 35, 'snowboard': 36, 'spoon': 50,
            'sports ball': 37, 'stop sign': 13, 'suitcase': 33, 'surfboard':
            42, 'teddy bear': 88, 'tennis racket': 43, 'tie': 32, 'toaster':
            80, 'toilet': 70, 'toothbrush': 90, 'traffic light': 10, 'train':
            7, 'truck': 8, 'tv': 72, 'umbrella': 28, 'vase': 86, 'wine glass':
            46, 'zebra': 24}
    
    
coco_fake_ids = { coco_ids[k]:i+1 for i,k in enumerate(sorted(coco_ids)) }    

coco_fake2real = { v:k for k,v in coco_fake_ids.items() }   
    
coco_ids_to_cats = dict(map(reversed, list(coco_ids.items())))


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


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class COCOLoader():
    cats_to_ids = dict(map(reversed, enumerate(COCO_CATS)))
    ids_to_cats = dict(enumerate(COCO_CATS))
    num_classes = len(COCO_CATS)
    categories = COCO_CATS[1:]   
    
    def __init__(self, root, annFile, included=[]):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.included_cats = included
        #self.ids  = self.get_ids()

        #self.ids = self.ids[:5000]
        ids = np.loadtxt("datasets/first5000.txt",dtype='int').tolist()
        self.ids = []
        i = 0 
        for img_id in ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)
            if target != []:
                self.ids.append(img_id)
                i+=1
            if i == 5000:
                break   
        print ("Loading 5000 images only...")

    def get_ids(self):
        all_ids = list(sorted(self.coco.imgs.keys()))
        finalset = set()
        if self.included_cats == []:
            return all_ids
        else:
            for cid in self.included_cats:
                if cid == 0:
                    continue
                actual_cid = coco_fake2real[cid]
                finalset = finalset.union(self.coco.catToImgs[actual_cid])
            return list(sorted(finalset))
        
    def show(self,image_id):
        import matplotlib.pyplot as plt
        index = self.ids.index(image_id)    
        _,target = self.__getitem__(index)
        path = self.coco.loadImgs(image_id)[0]['file_name']
        I = Image.open(os.path.join(self.root, path)).convert('RGB')
        plt.imshow(I)       
        for box, label in zip(target['boxes'],target['labels']): 
            xmin,ymin,xmax,ymax  = box.tolist()
            x = [xmin,ymin,xmax,ymax]
            rect = retbox(x)
            plt.plot(rect[:,0],rect[:,1],'r',linewidth = 2.0)
            plt.text(xmin,ymin,str(label.item()),fontsize=10,color='w',backgroundcolor='red')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
#        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = pil_loader(os.path.join(self.root, path))        
        H,W = img.height,img.width
        ann = self.convert(target)
        ann["size"] = torch.as_tensor([int(H),int(W)])
        return imgtransform(img), ann

    def __len__(self):
        return len(self.ids)

    def convert(self, target):
        boxes = []
        classes = []
        area = []
        iscrowd = []
        for obj in target:
            bbox = obj['bbox']
            xmin, ymin, w, h = bbox
            #is this right? 
            bbox = [xmin, ymin, w + xmin - 1 , h + ymin -1]
            cat = obj['category_id']  
            cat = coco_fake_ids[cat]
            difficult = int(obj['iscrowd'])
            if self.included_cats == [] or cat in self.included_cats:  
                if not difficult:
                    boxes.append(bbox)
                    classes.append(cat)
                    iscrowd.append(difficult)
                    area.append(w * h)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes).long()
        area = torch.as_tensor(area)
        iscrowd = torch.as_tensor(iscrowd)
        image_id = obj['image_id']
        image_id = torch.as_tensor([int(image_id)])
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target

#%%
if __name__ == '__main__':        
    DATASETS_ROOT = './datasets'
    split = 'val2014'
    root = '/home/manoj/%s' % (split)
    annFile = '%s/coco/annotations/instances_%s.json' % (DATASETS_ROOT,split)
    ld = COCOLoader(root,annFile,included=[1])
