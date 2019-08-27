# -*- coding: utf-8 -*-
"""generate chip from segmentation mask
"""

import os, sys
import cv2
import glob
import json
import argparse
import numpy as np
from tqdm import tqdm
from operator import add
import utils
import pdb

from datasets import get_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="show mask results")
    parser.add_argument('dataset', type=str, default='VisDrone',
                        choices=['VisDrone', 'HKB'], help='dataset name')
    parser.add_argument('--split', type=str, default='test',
                        help='dataset split (default: test)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    dataset = get_dataset(args.dataset)
    dest_datadir = dataset.detect_voc_dir
    image_dir = dest_datadir + '/JPEGImages'
    anno_dir = dest_datadir + '/Annotations'
    list_dir = dest_datadir + '/ImageSets/Main'
    mask_path = '../pytorch-deeplab-xception/run/mask-%s-%s' % (args.dataset.lower(), args.split)

    if not os.path.exists(dest_datadir):
        os.mkdir(dest_datadir)
        os.mkdir(image_dir)
        os.makedirs(list_dir)
        os.mkdir(anno_dir)

    test_list = dataset.get_imglist(split = args.split)

    chip_loc = {}
    chip_name_list = []
    for img_path in tqdm(test_list):
        imgid = os.path.basename(img_path)[:-4]
        origin_img = cv2.imread(img_path)
        mask_img = cv2.imread(os.path.join(mask_path, '%s.png'%imgid), cv2.IMREAD_GRAYSCALE)
        
        height, width = origin_img.shape[:2]
        mask_h, mask_w = mask_img.shape[:2]

        region_box, contours = utils.generate_box_from_mask(mask_img)
        if(len(region_box) == 0):
            print(img_path)
        region_box = utils.region_postprocess(region_box, contours, (mask_w, mask_h))
        region_box = utils.resize_box(region_box, (mask_w, mask_h), (width, height))
        region_box = utils.generate_crop_region(region_box, (width, height))

        for i, chip in enumerate(region_box):
            chip_img = origin_img[chip[1]:chip[3], chip[0]:chip[2], :].copy()
            chip_name = '%s_%s%d' % (imgid, args.split, i)
            cv2.imwrite(os.path.join(image_dir, '%s.jpg'%chip_name), chip_img)
            chip_name_list.append(chip_name)

            chip_info = {'loc': chip}
            chip_loc[chip_name] = chip_info

    # write test txt
    with open(os.path.join(list_dir, 'test.txt'), 'w') as f:
        f.writelines([x+'\n' for x in chip_name_list])
        print('write list txt, len=%d.' % len(chip_name_list))

    # write chip loc json
    with open(os.path.join(anno_dir, 'test_chip.json'), 'w') as f:
        json.dump(chip_loc, f, cls=utils.MyEncoder)
        print('write loc json.')