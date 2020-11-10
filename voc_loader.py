#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:27:16 2019

@author: manoj
"""

from scipy.io import loadmat
import cv2
import numpy as np
from PIL import Image
import torch
import collections
import os
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
    
    
    
from collections import defaultdict
import torchvision.transforms as transforms    

VOC_CATS = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
            'tvmonitor']

# DATASETS_ROOT = './datasets/'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')




class Loader(object):
    def __init__(self):
        pass

    def convert_and_maybe_resize(self, im, resize):
        scale = 1.0
        im = np.asarray(im)
        if resize:
            h, w, _ = im.shape
            scale = min(1000//max(h, w), 600//min(h, w))
            im = cv2.resize(im, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(im), scale
    
import transforms as T
def get_transform(istrain=False):
     transforms = []
     transforms.append(T.ToTensor())
     if istrain:
         transforms.append(T.RandomHorizontalFlip(0.5))
     return T.Compose(transforms)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
imgtransform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])



class VOC(Loader):

    def __init__(self, year, proposals, split, num_proposals=2000, included=[],
                 cats=VOC_CATS, root='./datasets/'):
        super().__init__()
        assert year in ['07', '12']
        self.CLASSES = VOC_CATS
        self.dataset = 'voc'
        self.year = year
        self.root = root + ('voc/VOCdevkit/VOC20%s/' % year)
        self.split = split
        assert split in ['train', 'val', 'trainval', 'test']
        self.proposals = proposals
        self.num_proposals = num_proposals
        assert num_proposals >= 0
        self.included_cats = included

        self.cats_to_ids = dict(map(reversed, enumerate(cats)))
        self.ids_to_cats = dict(enumerate(cats))
        self.num_classes = len(cats)
        self.categories = cats[1:]
        
        image_dir = os.path.join(self.root, 'JPEGImages')
        annotation_dir = os.path.join(self.root, 'Annotations') 
        
        file_names  = self.get_filenames()
        
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))       

    def __len__(self):
#        return 50
        return len(self.images)
       
    
    def class_stats(self):
        ids = defaultdict(set)
        for name in self.get_split_filenames():
            gt_cats = self.read_annotations(name)['labels']
            for cid in gt_cats:
                ids[int(cid)].add(name)
        return ids

    def show_stats(self):
        total = 0
        ids = self.class_stats()
        for i in ids.keys():
            print("%s: %i" % (VOC_CATS[i], len(ids[i])))
            total += len(ids[i])
        print("TOTAL: %i" % total)
            

    def load_image(self, name, resize=True):
        filepath = '{}JPEGImages/{}.jpg'.format(self.root, name)
        im = Image.open(filepath).convert('RGB')
        out,scale = self.convert_and_maybe_resize(im, resize)
        return out


    def get_filenames(self):
        all_files = self.get_split_filenames()
        finalset = set()
        if self.included_cats == []:
            return all_files
        else:
            ids = self.class_stats()
            for cid in self.included_cats:
                finalset = finalset.union(ids[cid])
            return list(finalset)
            
    def get_split_filenames(self):               
        with open(self.root+'ImageSets/Main/%s.txt' % self.split, 'r') as f:
            return f.read().split('\n')[:-1]

    def read_proposals(self, name):
        if self.proposals == 'edgeboxes':
            mat = loadmat('%sEdgeBoxesProposals/%s.mat' % (self.root, name))
            bboxes = mat['bbs'][:, :4]
        if self.proposals == 'selective_search':
            bboxes = np.load('%sSelectiveSearchProposals/%s.npy' % (self.root, name))
        if self.num_proposals == 0:
            return bboxes
        else:
            return bboxes[:self.num_proposals]


    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


    def convert(self, target):
        anno = target['annotation']
        H, W = anno['size']['height'], anno['size']['width']
        boxes = []
        classes = []
        area = []
        iscrowd = []
        objects = anno['object']
        if not isinstance(objects, list):
            objects = [objects]
        for obj in objects:
            bbox = obj['bndbox']
            bbox = [int(bbox[n]) - 1 for n in ['xmin', 'ymin', 'xmax', 'ymax']]
            cat = self.CLASSES.index(obj['name'])
            difficult = int(obj['difficult'])
            
            if self.included_cats == [] or cat in self.included_cats:   
                if not difficult:
                    boxes.append(bbox)
                    classes.append(cat)
                    iscrowd.append(difficult)
                    area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes)
        area = torch.as_tensor(area)
        iscrowd = torch.as_tensor(iscrowd)

        image_id = anno['filename'][0:6]
        image_id = torch.as_tensor([int(image_id)])

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes.long()
        target["image_id"] = image_id
        target["size"] = torch.as_tensor([int(H),int(W)])

        # for conversion to coco api
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target


    def __getitem__(self, index):       
        img = pil_loader(self.images[index])
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())
        return imgtransform(img), self.convert(target)


    def read_annotations(self, name):        
        tree = ET.parse('%sAnnotations/%s.xml' % (self.root, name))
        target = self.parse_voc_xml(tree.getroot())
        return self.convert(target)
        

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    loader = VOC('07', 'selective_search', 'trainval')
    loader1 = VOC('07', 'selective_search', 'trainval',included=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    l1 = set(loader1.get_filenames())
    for nextclass in range(16,21):
        loader2 = VOC('07', 'edgeboxes', 'trainval',included=[nextclass])
        l2 = set(loader2.get_filenames())
        intr = l1.intersection(l2)
        print (nextclass,'|',len(intr))


