#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 19:35:30 2019

@author: manoj
"""

import torch
import torch.nn as nn
import math
from torchvision.models.detection.image_list import ImageList

# min_size=800, max_size=1333
def batch_images(images, size_divisible=32):
    # concatenate
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    stride = size_divisible
    max_size = list(max_size)
    max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
    max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)
    max_size = tuple(max_size)

    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).zero_()
    for img, pad_img in zip(images, batched_imgs):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    return batched_imgs


def get_chopped(core_model):
    #core_model = get_model_FRCNN(num_classes = 11)
    
    lastblock = list(core_model.backbone.named_children())[-1][1]    
    d = list(lastblock[0].children())
    remove_c = d[3:7] #index of where we took feats from     
    #block1 = nn.Sequential(*remove_c)
    #block2 = nn.Sequential(*list(lastblock[1].children()))
    #block3 = nn.Sequential(*list(lastblock[2].children()))    
    allb = remove_c + list(lastblock[1].children()) + list(lastblock[2].children())    
    allb_s = nn.Sequential(*allb)
    return allb_s


#%%

choped_model = get_chopped(core_model)

a = torch.rand(512,38,34)
b = torch.rand(512,25,38)
images = [a,b]
#if image.dim() != 3:
#    raise ValueError("images is expected to be a list of 3d tensors "
#                     "of shape [C, H, W], got {}".format(image.shape))
image_sizes = [img.shape[-2:] for img in images]
images_res = batch_images(images)

features = choped_model(images_res)
targets = {}
images.image_sizes = []
proposals = []
core_model.roi_heads
detections, detector_losses = core_model.roi_heads(features, proposals, images.image_sizes, targets)
#detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)


#box_features = self.box_roi_pool(features, proposals, image_shapes)
#box_features = self.box_head(box_features)
#class_logits, box_regression = self.box_predictor(box_features)


#%%

#
#class ChoppedBackBone(nn.Module):
#    def __init(self):
#        super().__init__()
#        self.choped_model = get_chopped(core_model)
#        self.roi_heads = core_mode.roi_heads
#    def forward(self,images):
#                
#        





