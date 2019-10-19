import cv2
import random
import os, sys
import glob
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

import utils
from datasets import get_dataset

import pdb
import traceback

def parse_args():
    parser = argparse.ArgumentParser(description="analysis bbox size.")
    parser.add_argument('dataset', type=str, default='VisDrone',
                        choices=['VisDrone', 'HKB'], help='dataset name')
    args = parser.parse_args()
    return args

def parse_xml(file):
    xml = ET.parse(file).getroot()
    box_all = []
    pts = ['xmin', 'ymin', 'xmax', 'ymax']

    # size
    size = xml.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # bounding boxes
    for obj in xml.iter('object'):
        bbox = obj.find('bndbox')
        
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text) - 1
            bndbox.append(cur_pt)
        box_all += [bndbox]
    return box_all, (width, height)

if __name__ == '__main__':
    args = parse_args()
    dataset = get_dataset(args.dataset)

    origin_bboxsize = []
    train_list = dataset.get_imglist(split='train')
    for img_path in tqdm(train_list, ncols=80):
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        label, _ = dataset.get_gtbox(img_path)
        for box in label:
            origin_bboxsize.append((box[2] - box[0]) / width*1920)

    # after_bboxsize = []
    # with open(os.path.join(list_dir, 'train.txt'), 'r') as f:
    #     train_list = [x.strip() for x in f.readlines()]
    # for img_id in tqdm(train_list, ncols=80):
    #     label, size = parse_xml(os.path.join(anno_dir, img_id+'.xml'))
    #     for box in label:
    #         after_bboxsize.append((box[2] - box[0]) / size[0])

    bins = [x*5+20 for x in range(0, 20)]
    plt.hist(origin_bboxsize, bins=bins, density=False, label='size', histtype='bar')
    # plt.hist(after_bboxsize, bins=bins, density=True, label='after', histtype='step')
    plt.legend()
    plt.xlabel('size')
    plt.ylabel('nums')

    plt.title(u'size show')

    plt.show()