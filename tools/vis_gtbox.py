"""
check train chips labels
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

dbroot = Path.db_root_dir()

src_traindir = dbroot + '/VisDrone2019-DET-train'
src_valdir = dbroot + '/VisDrone2019-DET-val'
src_testdir = dbroot + '/VisDrone2019-DET-test-challenge'

dest_datadir = dbroot + '/detect_voc'
image_dir = dest_datadir + '/JPEGImages'
anno_dir = dest_datadir + '/Annotations'
list_dir = dest_datadir + '/ImageSets/Main'

def parse_xml(file):
    xml = ET.parse(file).getroot()
    box_all = []
    pts = ['xmin', 'ymin', 'xmax', 'ymax']

    # size
    location = xml.find('location')
    loc = []
    for i, pt in enumerate(pts):
        cur_pt = int(location.find(pt).text) - 1
        loc.append(cur_pt)

    # bounding boxes
    for obj in xml.iter('object'):
        bbox = obj.find('bndbox')
        
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text) - 1
            bndbox.append(cur_pt)
        box_all += [bndbox]
    return box_all, loc

def get_box_label(img_path):
    anno_path = img_path.replace('images', 'annotations')
    anno_path = anno_path.replace('jpg', 'txt')
    with open(anno_path, 'r') as f:
        data = [x.strip().split(',')[:8] for x in f.readlines()]
        annos = np.array(data)

    boxes = annos[annos[:, 4] == '1'][:, :4].astype(np.int32)
    y = np.zeros_like(boxes)
    y[:, 0] = boxes[:, 0]
    y[:, 1] = boxes[:, 1]
    y[:, 2] = boxes[:, 0] + boxes[:, 2]
    y[:, 3] = boxes[:, 1] + boxes[:, 3]
    
    return y


def _boxvis(img, gt_box_list):
    img1 = img.copy()
    for box in gt_box_list:
        cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

    plt.subplot(1, 1, 1); plt.imshow(img1[:, :, [2,1,0]])
    plt.show()
    cv2.waitKey(0)

    
if __name__ == '__main__':
    train_list = glob.glob(src_traindir + '/images/*.jpg')

    with open(os.path.join(list_dir, 'train.txt'), 'r') as f:
        img_list = [x.strip() for x in f.readlines()]

    for img_path in train_list:
        img = cv2.imread(img_path)
        imgid = os.path.basename(img_path)[:-4]
        crop_list = []
        for crop_id in img_list:
            if imgid in crop_id:
                crop_list.append(crop_id)
        
        loc_list = []
        for name in crop_list:
            # crop_img = cv2.imread(os.path.join(image_dir, name+'.jpg'))
            boxes, loc = parse_xml(os.path.join(anno_dir, name+'.xml'))
            assert loc[2] - loc[0] == loc[3] - loc[1]
            loc_list.append(loc)
        # _boxvis(img, loc_list)
    