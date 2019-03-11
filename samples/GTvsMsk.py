#!/usr/bin/env python
# coding: utf-8
import os
import sys
import random
import skimage.io
import time

ROOT_DIR = '/home/tj816/Mask_RCNN-master'
sys.path.append(ROOT_DIR)
# Import Mask RCNN
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
import samples.coco.coco as coco
from train_test import DrugDataset


# Root directory of the project
PIC_NAME = "temporary_dataset/"
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
WEIGHT_DIR = "shapes20190307T1102/mask_rcnn_shapes_0050.h5"
IMAGE_DIR = os.path.join(ROOT_DIR, PIC_NAME+"cv2_mask")
GROUND_TRUTH_DIR = os.path.join(ROOT_DIR, PIC_NAME+'cv2_mask')
class_names = ['BG', 'leakage']

# 基础设置
dataset_root_path = os.path.join(ROOT_DIR, PIC_NAME)
img_floder = os.path.join(dataset_root_path, "pic")
mask_floder = os.path.join(dataset_root_path, "cv2_mask")
imglist = os.listdir(img_floder)
gt_num = len(imglist)

# Directory to save logs and trained model
COCO_MODEL_PATH = os.path.join(MODEL_DIR, WEIGHT_DIR)

# CocoConfig determine the number of layers of nets
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
# Run detection
count = os.listdir(IMAGE_DIR)
start_time = time.time()
dataset_test = DrugDataset()
dataset_test.load_shapes(gt_num,img_floder,mask_floder,imglist,dataset_root_path)
dataset_test.prepare()

for i in range(gt_num):
        image = skimage.io.imread(os.path.join(img_floder,imglist[i]))
        r=dataset_test[i]
        r = modellib.load_image_gt(dataset_test,config=config,image_id=i)
        # print(dataset_test)
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

# for i in range(len(count)):
#     path = os.path.join(IMAGE_DIR, count[i])
#     if os.path.isfile(path):
#         # file_names = next(os.walk(IMAGE_DIR))[2]
#         image = skimage.io.imread(path)
#         results = model.detect([image], verbose=0)
#         r = results[0]
#         t = modellib.load_image_gt()
#         visualize.display_differences(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

end_time = time.time()
print("总共检测时间= ", end_time - start_time)
print("总共检测图片数量", len(count))
print("平均检测时间= ", (end_time - start_time) / len(count))
