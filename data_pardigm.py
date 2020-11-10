#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:40:06 2019

@author: manoj
"""

from voc_loader import VOC

def train_test_full():

    num_classes = 21 # 20 classes + background for VOC
    classes = list(range(0,21))
    print ("{} classes: {}".format(num_classes,classes))
    dataset =    VOC('07', 'edgeboxes', 'trainval')
    dataset_test = VOC('07', 'edgeboxes', 'test')
        
    print('data prepared, train data: {}'.format(len(dataset)))
    print('data prepared, test data: {}'.format(len(dataset_test)))
    
    yield num_classes, dataset, dataset_test    


def train_test_half():

    num_classes = 11 # first 10 classes + background for VOC
    included = [1,2,3,4,5,6,7,8,9,10]
    print ("{} classes: {}".format(num_classes,included))
    dataset =    VOC('07', 'edgeboxes', 'trainval',included=included)
    dataset_test = VOC('07', 'edgeboxes', 'test',included=included)
        
    print('data prepared, train data: {}'.format(len(dataset)))
    print('data prepared, test data: {}'.format(len(dataset_test)))
    
    yield num_classes, dataset, dataset_test    

def half_batch():
    classes = [[0,1,2,3,4,5,6,7,8,9,10]] + [list(range(11,21))]
    included = []
    num_classes = 0
    for c in classes:
        included += c
        num_classes = len(included)
        print ("{} classes: {}".format(num_classes,c))
        dataset =    VOC('07', 'edgeboxes', 'trainval',included=c)
        dataset_test = VOC('07', 'edgeboxes', 'test',included=included)
            
        print('data prepared, train data: {}'.format(len(dataset)))
        print('data prepared, test data: {}'.format(len(dataset_test)))
    
        yield num_classes, dataset, dataset_test   

def half_offlines():
    classes = [ list(range(0,11))]  + [[i] for i in range(11,21)]
    included = []
    num_classes = 0
    for c in classes:
        included += c
        num_classes = len(included)
        print ("{} classes: {}".format(num_classes,included))
        dataset =    VOC('07', 'edgeboxes', 'trainval',included = included)
        dataset_test = VOC('07', 'edgeboxes', 'test',included = included)
            
        print('data prepared, train data: {}'.format(len(dataset)))
        print('data prepared, test data: {}'.format(len(dataset_test)))
    
        yield num_classes, dataset, dataset_test   



def half_incr():
    classes = [ list(range(0,11))]  + [[i] for i in range(11,21)]
    included = []
    num_classes = 0
    for c in classes:
        included += c
        num_classes = len(included)
        print ("{} classes: {}".format(num_classes,c))
        dataset =    VOC('07', 'edgeboxes', 'trainval',included=c)
        dataset_test = VOC('07', 'edgeboxes', 'test',included=included)
            
        print('data prepared, train data: {}'.format(len(dataset)))
        print('data prepared, test data: {}'.format(len(dataset_test)))
    
        yield num_classes, dataset, dataset_test   
    

data_dict = {}
data_dict['full_voc'] = train_test_full
data_dict['half_voc'] = train_test_half
data_dict['half_offlines'] = half_offlines
data_dict['half_finteune'] = half_offlines
data_dict['halfbatch_voc'] = half_batch
data_dict['incr_voc'] = half_incr
