"""
vis mask
"""

import os
import sys
import cv2
import glob
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
import xml.etree.ElementTree as ET
import pdb
import utils

from datasets import get_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="show mask results")
    parser.add_argument('dataset', type=str, default='VisDrone',
                        choices=['VisDrone', 'HKB'], help='dataset name')
    args = parser.parse_args()
    return args

def _vis(img_path, dataset):
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    img_id = os.path.basename(img_path)[:-4]
    pred_mask_path = os.path.join(pred_mask_dir, img_id+'.png')
    label_mask_path = os.path.join(segmentation_dir, img_id+'_region.png')
    pred_mask = cv2.imread(pred_mask_path) * 255
    label_mask = cv2.imread(label_mask_path) * 255

    # bounding box
    img1 = img.copy()
    gt_box_list, _ = dataset.get_gtbox(img_path)
    for box in gt_box_list:
        cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    #     cv2.putText(img1, str((box[2], box[3])), (box[0], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
    # cv2.putText(img1, str((width, height)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
    # label_region_box, _ = utils.generate_box_from_mask(label_mask[:, :, 0])
    # label_region_box = utils.resize_box(label_region_box, (40, 30), (width, height))
    # for box in label_region_box:
    #     cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 5)

    # region box
    img2 = img.copy()
    mask_h, mask_w = pred_mask.shape[:2]
    region_box, contours = utils.generate_box_from_mask(pred_mask[:, :, 0])
    resize_region_box = utils.resize_box(region_box, (mask_w, mask_h), (width, height))
    for box in resize_region_box:
        cv2.rectangle(img2, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

    # region postprocess
    img3 = img.copy()
    new_regions = utils.region_postprocess(region_box, contours, (mask_w, mask_h))
    resize_region_box = utils.resize_box(new_regions, (mask_w, mask_h), (width, height))
    # new_regions = utils.generate_crop_region(resize_region_box, (width, height))
    for box in resize_region_box:
        cv2.rectangle(img3, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

    img4 = img.copy()
    # resize_region_box = utils.resize_box(temp, (mask_w, mask_h), (width, height))
    new_regions = utils.generate_crop_region(resize_region_box, (width, height))
    for box in new_regions:
        cv2.rectangle(img4, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

    plt.subplot(2, 3, 1); plt.imshow(img1[:, :, [2,1,0]])
    plt.subplot(2, 3, 2); plt.imshow(img2[:, :, [2,1,0]])
    plt.subplot(2, 3, 3); plt.imshow(img3[:, :, [2,1,0]])
    plt.subplot(2, 3, 4); plt.imshow(label_mask[:, :, [2,1,0]])
    plt.subplot(2, 3, 5); plt.imshow(pred_mask[:, :, [2,1,0]])
    plt.subplot(2, 3, 6); plt.imshow(img4[:, :, [2,1,0]])

    dirname = os.path.join(pred_mask_dir, os.path.pardir)
    cv2.imwrite(os.path.join(dirname, 'virtualization', 'image.jpg'), img)
    cv2.imwrite(os.path.join(dirname, 'virtualization', 'bbox.jpg'), img1)
    cv2.imwrite(os.path.join(dirname, 'virtualization', 'brec.jpg'), img2)
    cv2.imwrite(os.path.join(dirname, 'virtualization', 'post_process.jpg'), img3)
    cv2.imwrite(os.path.join(dirname, 'virtualization', 'label_mask.jpg'), label_mask)
    cv2.imwrite(os.path.join(dirname, 'virtualization', 'pred_mask.jpg'), pred_mask)
    cv2.imwrite(os.path.join(dirname, 'virtualization', 'result.jpg'), img4)

    
    plt.show()
    cv2.waitKey(0)

    
if __name__ == '__main__':
    args = parse_args()
    dataset = get_dataset(args.dataset)
    dest_datadir = dataset.region_voc_dir
    image_dir = dest_datadir + '/JPEGImages'
    segmentation_dir = dest_datadir + '/SegmentationClass'
    list_folder = dest_datadir + '/ImageSets'

    pred_mask_dir = '../pytorch-deeplab-xception/run/mask-%s-val' % args.dataset.lower()
    val_list = dataset.get_imglist('val')

    for img_path in val_list[2:]:
        _vis(img_path, dataset)
    