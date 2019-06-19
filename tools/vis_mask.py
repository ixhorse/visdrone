"""
vis mask
"""

import os
import sys
import cv2
import glob
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
import xml.etree.ElementTree as ET
from path import Path
import pdb
import utils

dbroot = Path.db_root_dir()

src_traindir = dbroot + '/VisDrone2019-DET-train'
src_valdir = dbroot + '/VisDrone2019-DET-val'
src_testdir = dbroot + '/VisDrone2019-DET-test-challenge'

dest_datadir = dbroot + '/region_voc'
image_dir = dest_datadir + '/JPEGImages'
segmentation_dir = dest_datadir + '/SegmentationClass'
list_folder = dest_datadir + '/ImageSets'

pred_mask_dir = '../pytorch-deeplab-xception/run/mask'


def _vis(img_path):
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    img_id = os.path.basename(img_path)[:-4]
    pred_mask_path = os.path.join(pred_mask_dir, img_id+'.png')
    label_mask_path = os.path.join(segmentation_dir, img_id+'_region.png')
    pred_mask = cv2.imread(pred_mask_path) * 255
    label_mask = cv2.imread(label_mask_path) * 255

    # bounding box
    img1 = img.copy()
    gt_box_list = utils.get_box_label(img_path)
    for box in gt_box_list:
        cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 4)

    # region box
    img2 = img.copy()
    mask_h, mask_w = pred_mask.shape[:2]
    region_box = utils.generate_box_from_mask(pred_mask[:, :, 0])
    region_box = utils.resize_box(region_box, (mask_w, mask_h), (width, height))
    for box in region_box:
        cv2.rectangle(img2, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 4)

    # region postprocess
    img3 = img.copy()
    new_regions = utils.region_postprocess(region_box, (width, height))
    new_regions = utils.generate_crop_region(new_regions, (width, height))
    for box in new_regions:
        cv2.rectangle(img3, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 4)

    plt.subplot(2, 3, 1); plt.imshow(img1[:, :, [2,1,0]])
    plt.subplot(2, 3, 2); plt.imshow(img2[:, :, [2,1,0]])
    plt.subplot(2, 3, 3); plt.imshow(img3[:, :, [2,1,0]])
    plt.subplot(2, 3, 4); plt.imshow(label_mask[:, :, [2,1,0]])
    plt.subplot(2, 3, 5); plt.imshow(pred_mask[:, :, [2,1,0]])
    
    plt.show()
    cv2.waitKey(0)

    
if __name__ == '__main__':
    val_list = glob.glob(src_valdir + '/images/*.jpg')

    for img_path in val_list:
        _vis(img_path)
    