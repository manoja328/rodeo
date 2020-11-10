#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:39:07 2019

@author: manoj
"""

#%%
import xml.etree.ElementTree as ET
from collections import defaultdict
from data_pardigm import half_incr
from voc_loader import VOC_CATS,VOC
#%%
# stats of how many objets in each iteration

for c,ds,ds_test in half_incr():
    class_stats = defaultdict(int)
   
    for file in ds.annotations:
        name = file.split("/")[-1].replace(".xml","")
        vv = ds.parse_voc_xml(ET.parse(file).getroot())
        anns = vv['annotation']['object']
        if isinstance(anns,dict):
            anns = [anns]
        for ann in anns:
            lab = ann['name']
            if lab not in VOC_CATS:
                print (lab)
            if ann['difficult'] == '0':
                class_stats[lab] +=1
    for key in VOC_CATS:
        print("{}".format(class_stats.get(key,0)),end=',')

    

#%%
# are there any single object images
loader = VOC('07', 'selective_search', 'trainval',included=[19])        
a = {}
for index in range(len(loader)):
    r = loader.parse_voc_xml(ET.parse(loader.annotations[index]).getroot())
    a[index] = r['annotation']['object']        
        