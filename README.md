# RODEO: Replay for Online Object Detection

This is an official implementation of our **BMVC 2020 paper RODEO: Replay for Online Object Detection.** The arxiv link of the paper is available [here](https://arxiv.org/abs/2008.06439).
## Abstract
Humans can incrementally learn to do new visual detection tasks, which is a huge challenge for today's computer vision systems. Incrementally trained deep learning models lack backwards transfer to previously seen classes and suffer from a phenomenon known as ``catastrophic forgetting.`` In this paper, we pioneer online streaming learning for object detection, where an agent must learn examples one at a time with severe memory and computational constraints. In object detection, a system must output all bounding boxes for an image with the correct label. Unlike earlier work, the system described in this paper can learn how to do this task in an online manner with new classes being introduced over time. We achieve this capability by using a novel memory replay mechanism that replays entire scenes in an efficient manner. We achieve state-of-the-art results on both the PASCAL VOC 2007 and MS COCO datasets.

![RODEO](https://raw.githubusercontent.com/manoja328/manoja328.github.io/master/assets/rodeo.jpg)



## Dependencies
We recommend setting up a conda environment with the envrionment file in this repo.

```conda env create --file envname.yml```

## Setup VOC 2007 and MSCOCO-214 datasets

To setup the dataset and evaluation donwload COCO API as suggested in [pytorch object detection tuturial.](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) and this [colab notebook](https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb).
We use pre-calcuated Edgebox proposals to train our Fast-RCNN models.
### For VOC dataset, the half checkpoint and Edgebox proposals are located in [this google drive folder.](https://drive.google.com/drive/folders/1APMtR7vv4onTidW6Rd5OdpivXrGjlyTW?usp=sharing)


## To train RODEO:
1. To train the models for base init classes run `train_bettercoco.py` or `train_better` for VOC. This outputs the half checkpoint that can be used to incremental training.
2. To extract features run `extract_coco_features.py` or `extract_voc_features.py` file which will output features to h5 files.
3. To get the PQ reconstrcuted features run `pq_encoding_voc.py` or `pq_encoding_coco.py`.
4. Run `runcoco.sh` to train offline and other baseline models.
5. Run `runreplay.sh` to run RODEO model with replay sizes as a hyper-parameter.



## References
One of the comparison method (ILWFOD) discussed is proposed in ICCV 2017 paper ["Incremental Learning of Object Detectors without Catastrophic Forgetting"](https://arxiv.org/abs/1708.06977). Their code can be found in this repo https://github.com/kshmelkov/incremental_detectors.


## Citation
If using this code, please cite our paper.

```
@inproceedings{acharya2020rodeo,
title={RODEO: Replay for Online Object Detection},
author={Acharya, Manoj and Hayes, Tyler L. and Kanan, Christopher},
booktitle={The British Machine Vision Conference},
year={2020}
}
```
