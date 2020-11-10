#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:28:12 2019

@author: manoj
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def distill_bbLoss(bbox,target):
    #box_loss torch.sum((bbox - target)**2)
    #box_loss = F.smooth_l1_loss(bbox,target,reduction="sum")
    box_loss = F.mse_loss(bbox,target,reduction="mean")
    return box_loss

def distill_CELoss(logits,target):
    #subtract mean over class dimension from un-normalized logits
    logits = logits - torch.mean(logits,dim = 0, keepdim=True)
    target = target - torch.mean(target,dim = 0, keepdim=True)
    #class_distillation_loss = torch.sum((logits - target)**2)
    class_distillation_loss =  F.mse_loss(logits,target,reduction="mean")
    return class_distillation_loss
 
ce_loss_fct = nn.KLDivLoss(reduction='batchmean')    
def distill_CELoss_hinton(logits,target):
    temperature = 1

    loss_ce = ce_loss_fct(F.log_softmax(logits/temperature, dim=-1),
                        F.softmax(target/temperature, dim=-1)) * (temperature)**2
    return loss_ce


#def distillation_loss(old, new):
#    
#    distillation_logits , distillation_boxes = old
#    current_logits , current_boxes = new
#    
#    celoss = distill_CELoss(current_logits,distillation_logits)
#    bbloss = distill_bbLoss(current_boxes,distillation_boxes)
#    
##    losses = {
##        "distill_loss_classifier": celoss,
##        "distill_loss_bbox": bbloss
##    }
##    
#    return celoss + bbloss
    
    

#
#    classification_loss = F.cross_entropy(class_logits, labels)
#
#    # get indices that correspond to the regression targets for
#    # the corresponding ground truth labels, to be used with
#    # advanced indexing
#    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
#    labels_pos = labels[sampled_pos_inds_subset]
#    N, num_classes = class_logits.shape
#    box_regression = box_regression.reshape(N, -1, 4)
#
#    box_loss = F.smooth_l1_loss(
#        box_regression[sampled_pos_inds_subset, labels_pos],
#        regression_targets[sampled_pos_inds_subset],
#        reduction="sum",
#    )
#    box_loss = box_loss / labels.numel()
#
#    return classification_loss, box_loss