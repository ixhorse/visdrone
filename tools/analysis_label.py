import cv2
import random
import os, sys
import glob
import numpy as np
from tqdm import tqdm
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
    label_all = []
    pts = ['xmin', 'ymin', 'xmax', 'ymax']
    # bounding boxes
    for obj in xml.iter('object'):
        label_all.append(int(obj.find('name').text))

        bbox = obj.find('bndbox')
        
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text) - 1
            bndbox.append(cur_pt)
        box_all += [bndbox]
    return box_all, label_all


if __name__ == '__main__':
    train_list = glob.glob(src_traindir + '/images/*.jpg')

    label_count = {i+1 : 0 for i in range(10)}
    for img_path in tqdm(train_list, ncols=80):
        bboxes, labels = utils.get_box_label(img_path)
        for cid in labels:
            label_count[cid] += 1
    print(label_count)

    label_count = {i+1 : 0 for i in range(10)}
    with open(os.path.join(list_dir, 'train.txt'), 'r') as f:
        train_list = [x.strip() for x in f.readlines()]
    for img_id in tqdm(train_list, ncols=80):
        bboxes, labels = parse_xml(os.path.join(anno_dir, img_id+'.xml'))
        for cid in labels:
            label_count[cid] += 1
    print(label_count)