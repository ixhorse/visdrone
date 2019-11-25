import os, sys
import cv2
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import utils
import pdb

from datasets import get_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="show mask results")
    parser.add_argument('dataset', type=str, default='VisDrone',
                        choices=['VisDrone', 'HKB'], help='dataset name')
    args = parser.parse_args()
    return args   

if __name__ == '__main__':
    args = parse_args()
    dataset = get_dataset(args.dataset)
    val_list = dataset.get_imglist('val')
    mask_path = '../pytorch-deeplab-xception/run/mask-%s-val' % args.dataset.lower()

    label_object = []
    detect_object = []
    mask_object = []
    undetected_img = []
    pixel_num = []
    for img_path in tqdm(val_list, ncols=80):
        img_name = os.path.basename(img_path)
        raw_file = os.path.join(mask_path, img_name[:-4]+'.png')
        
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        mask_img = cv2.imread(raw_file, cv2.IMREAD_GRAYSCALE)
        mask_h, mask_w = mask_img.shape[:2]
        
        pixel_num.append(np.sum(mask_img))

        label_box, _ = dataset.get_gtbox(img_path)
        region_box, contours = utils.generate_box_from_mask(mask_img)
        region_box = utils.region_postprocess(region_box, contours, (mask_w, mask_h))
        region_box = utils.resize_box(region_box, (mask_w, mask_h), (width, height))
        region_box = utils.generate_crop_region(region_box, (width, height))

        count = 0
        for box1 in label_box:
            for box2 in region_box:
                if utils.overlap(box2, box1):
                    count += 1
                    break

        label_object.append(len(label_box))
        detect_object.append(count)
        mask_object.append(len(region_box))
        if len(label_box) != count:
            undetected_img.append(img_name)

    print('recall: %f' % (np.sum(detect_object) / np.sum(label_object)))
    # print('cost avg: %f, std: %f' % (np.mean(pixel_num), np.std(pixel_num)))
    print('detect box avg: %f' %(np.mean(mask_object)))
    # print(sorted(undetected_img))