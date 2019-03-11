#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN - Train on Shapes Dataset
#
#
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
#
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster.


import os
import skimage
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

ROOT_DIR = "/home/tj816/Mask_RCNN-master"
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/shapes20190307T1102/mask_rcnn_shapes_0050.h5")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
class_names = ['BG', 'leakage']

sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from train_test import ShapesConfig, DrugDataset

config = ShapesConfig()
config.display()


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

model.load_weights(COCO_MODEL_PATH, by_name=True)
PIC_NAME = "temporary_dataset/"
dataset_root_path = os.path.join(ROOT_DIR, PIC_NAME)
img_floder = os.path.join(dataset_root_path, "pic")
mask_floder = os.path.join(dataset_root_path, "cv2_mask")
imglist = os.listdir(img_floder)
gt_num = len(imglist)

# Validation dataset
dataset_test = DrugDataset()
dataset_test.load_shapes(gt_num, img_floder, mask_floder, imglist, dataset_root_path)
dataset_test.prepare()

for i in range(gt_num):
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_test, inference_config,
                                                                                       i, use_mini_mask=False)

    results = model.detect([original_image], verbose=0)
    r = results[0]

    visualize.display_differences(imglist[i], original_image, gt_bbox, np.asarray([1]), gt_mask, r['rois'],
                                  np.asarray([1]), r['scores'], r['masks'], class_names)
