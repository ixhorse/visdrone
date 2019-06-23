# -*- coding: utf-8 -*-
"""generate chip from segmentation mask
"""

import os, sys
import cv2
import glob
import json
import random
import numpy as np
from tqdm import tqdm
from operator import add
from path import Path
import utils
import pdb

dbroot = Path.db_root_dir()

src_traindir = dbroot + '/VisDrone2019-DET-train'
src_valdir = dbroot + '/VisDrone2019-DET-val'
src_testdir = dbroot + '/VisDrone2019-DET-test-challenge'

dest_datadir = dbroot + '/detect_voc'
image_dir = dest_datadir + '/JPEGImages'
anno_dir = dest_datadir + '/Annotations'
list_dir = dest_datadir + '/ImageSets/Main'

mask_path = '../pytorch-deeplab-xception/run/mask'

def main():
    if not os.path.exists(dest_datadir):
        os.mkdir(dest_datadir)
        os.mkdir(image_dir)
        os.makedirs(list_dir)
        os.mkdir(anno_dir)

    test_list = glob.glob(src_testdir + '/images/*.jpg')
    test_list = [os.path.basename(x)[:-4] for x in test_list]

    chip_loc = {}
    chip_name_list = []
    for imgid in tqdm(test_list):
        origin_img = cv2.imread(os.path.join(src_testdir, 'images', '%s.jpg'%imgid))
        mask_img = cv2.imread(os.path.join(mask_path, '%s.png'%imgid), cv2.IMREAD_GRAYSCALE)
        
        height, width = origin_img.shape[:2]
        mask_h, mask_w = mask_img.shape[:2]

        region_box = utils.generate_box_from_mask(mask_img)
        region_box = utils.resize_box(region_box, (mask_w, mask_h), (width, height))
        region_box = utils.region_postprocess(region_box, (width, height))
        region_box = utils.generate_crop_region(region_box, (width, height))

        for i, chip in enumerate(region_box):
            chip_img = origin_img[chip[1]:chip[3], chip[0]:chip[2], :].copy()
            chip_name = '%s_%d' % (imgid, i)
            cv2.imwrite(os.path.join(image_dir, '%s.jpg'%chip_name), chip_img)
            chip_name_list.append(chip_name)

            chip_info = {'loc': chip}
            chip_loc[chip_name] = chip_info

    # write test txt
    with open(os.path.join(list_dir, 'test.txt'), 'w') as f:
        f.writelines([x+'\n' for x in chip_name_list])
        print('write txt.')

    # write chip loc json
    with open(os.path.join(anno_dir, 'test_chip.json'), 'w') as f:
        json.dump(chip_loc, f)
        print('write json.')

if __name__ == '__main__':
    main()