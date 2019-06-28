import cv2
import random
import os, sys
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from path import Path
import utils

import pdb
import traceback

dbroot = Path.db_root_dir()

src_traindir = dbroot + '/VisDrone2019-DET-train'
src_valdir = dbroot + '/VisDrone2019-DET-val'
src_testdir = dbroot + '/VisDrone2019-DET-test-challenge'

voc_datadir = dbroot + '/detect_voc'
image_dir = voc_datadir + '/JPEGImages'
anno_dir = voc_datadir + '/Annotations'
list_dir = voc_datadir + '/ImageSets/Main'

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
    origin_bboxsize = []
    train_list = glob.glob(src_traindir + '/images/*.jpg')
    for img_path in tqdm(train_list, ncols=80):
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        label, _ = utils.get_box_label(img_path)
        for box in label:
            origin_bboxsize.append((box[2] - box[0]) / width)

    after_bboxsize = []
    with open(os.path.join(list_dir, 'train.txt'), 'r') as f:
        train_list = [x.strip() for x in f.readlines()]
    for img_id in tqdm(train_list, ncols=80):
        label, size = parse_xml(os.path.join(anno_dir, img_id+'.xml'))
        for box in label:
            after_bboxsize.append((box[2] - box[0]) / size[0])

    bins = [0., 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.3, 0.4, 0.5, 1]
    plt.hist(origin_bboxsize, bins=bins, density=True, label='origin', alpha=0.4, histtype='step')
    plt.hist(after_bboxsize, bins=bins, density=True, label='after', alpha=0.4, histtype='step')
    plt.legend()
    plt.xlabel('size')
    plt.ylabel('count')

    plt.title(u'size show')

    plt.show()