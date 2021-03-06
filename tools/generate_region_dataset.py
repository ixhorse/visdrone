"""convert VOC format
+ region_voc
    + JPEGImages
    + SegmentationClass
"""

import os, sys
import glob
import cv2
import random
import shutil
import argparse
import numpy as np
import pandas as pd
import concurrent.futures
import pdb

from datasets import get_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="convert to voc dataset")
    parser.add_argument('dataset', type=str, default='VisDrone',
                        choices=['TT100K', 'VisDrone', 'HKB'], help='dataset name')
    parser.add_argument('mode', type=str, default='train',
                        choices=['train', 'test'], help='for train or test')
    args = parser.parse_args()
    return args

# copy train and test images
def _copy(src_image, dest_path):
    shutil.copy(src_image, dest_path)


def _resize(src_image, dest_path):
    img = cv2.imread(src_image)

    height, width = img.shape[:2]
    size = (1024, 1024)

    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    name = os.path.basename(src_image)
    cv2.imwrite(os.path.join(dest_path, name), img)

def _myaround_up(value):
    tmp = np.floor(value).astype(np.int32)
    return tmp + 1 if value - tmp > 0.05 else tmp

def _myaround_down(value):
    tmp = np.ceil(value).astype(np.int32)
    return max(0, tmp - 1 if tmp - value > 0.05 else tmp)

def _generate_mask(img_path, dataset):
    try:
        # image mask
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        
        # chip mask 40x30, model input size 640x480
        mask_w, mask_h = 30, 30

        region_mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
        boxes, _ = dataset.get_gtbox(img_path)
        # for box in boxes:
        #     xmin = np.floor(1.0 * box[0] / width * mask_w).astype(np.int32)
        #     ymin = np.floor(1.0 * box[1] / height * mask_h).astype(np.int32)
        #     xmax = np.floor(1.0 * box[2] / width * mask_w).astype(np.int32)
        #     ymax = np.floor(1.0 * box[3] / height * mask_h).astype(np.int32)
        #     ignore_xmin = xmin - 1 if xmin - 1 >= 0 else 0
        #     ignore_ymin = ymin - 1 if ymin - 1 >= 0 else 0
        #     ignore_xmax = xmax + 1 if xmax + 1 < mask_w else mask_w - 1
        #     ignore_ymax = ymax + 1 if ymax + 1 < mask_h else mask_h - 1
        #     region_mask[ignore_ymin : ignore_ymax+1, ignore_xmin : ignore_xmax+1] = 255
        for box in boxes:
            xmin = _myaround_down(1.0 * box[0] / width * mask_w)
            ymin = _myaround_down(1.0 * box[1] / height * mask_h)
            xmax = _myaround_up(1.0 * box[2] / width * mask_w)
            ymax = _myaround_up(1.0 * box[3] / height * mask_h)
            region_mask[ymin : ymax, xmin : xmax] = 1
        maskname = os.path.join(segmentation_dir, img_name[:-4] + '_region.png')
        cv2.imwrite(maskname, region_mask)

    except Exception as e:
        print(e)
        print(img_path)


if __name__ == "__main__":
    args = parse_args()
    print(args)

    dataset = get_dataset(args.dataset)
    dest_datadir = dataset.region_voc_dir
    image_dir = dest_datadir + '/JPEGImages'
    segmentation_dir = dest_datadir + '/SegmentationClass'
    annotation_dir = dest_datadir + '/Annotations'
    list_folder = dest_datadir + '/ImageSets'

    if not os.path.exists(dest_datadir):
        os.mkdir(dest_datadir)
        os.mkdir(image_dir)
        os.mkdir(segmentation_dir)
        os.mkdir(annotation_dir)
        os.mkdir(list_folder)
  
    if 'train' in args.mode:
        train_list = dataset.get_imglist('train')
        val_list = dataset.get_imglist('val')
        trainval_list = train_list + val_list
        print(len(train_list), len(val_list))

        with open(os.path.join(list_folder, 'train.txt'), 'w') as f:
            temp = [os.path.basename(x)[:-4]+'\n' for x in train_list]
            f.writelines(temp)
        with open(os.path.join(list_folder, 'val.txt'), 'w') as f:
            temp = [os.path.basename(x)[:-4]+'\n' for x in val_list]
            f.writelines(temp)
        
        print('copy train_val images....')
        with concurrent.futures.ThreadPoolExecutor() as exector:
            exector.map(_resize, trainval_list, [image_dir]*len(trainval_list))

        print('generate masks...')
        with concurrent.futures.ThreadPoolExecutor() as exector:
            exector.map(_generate_mask, trainval_list, [dataset]*len(trainval_list))
        
        if args.dataset == 'HKB':
            print('copy box annos...')
            train_anno_list = dataset.get_annolist('train')
            val_anno_list = dataset.get_annolist('val')
            trainval_anno_list = train_anno_list + val_anno_list
            with concurrent.futures.ThreadPoolExecutor() as exector:
                exector.map(_copy, trainval_anno_list, [annotation_dir]*len(trainval_anno_list))

        print('done.')    

    if 'test' in args.mode:
        test_list = dataset.get_imglist('test')
        with open(os.path.join(list_folder, 'test.txt'), 'w') as f:
            temp = [os.path.basename(x)[:-4]+'\n' for x in test_list]
            f.writelines(temp)
        print('copy test image....')
        with concurrent.futures.ThreadPoolExecutor() as exector:
            exector.map(_copy, test_list, [image_dir]*len(test_list))