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

dest_datadir = dbroot + '/region_voc'
image_dir = dest_datadir + '/JPEGImages'
segmentation_dir = dest_datadir + '/SegmentationClass'
list_folder = dest_datadir + '/ImageSets'

def parse_xml(file):
    xml = ET.parse(file).getroot()
    box_all = []
    pts = ['xmin', 'ymin', 'xmax', 'ymax']

    # size
    location = xml.find('location')
    width = int(location.find('xmax').text) - int(location.find('xmin').text)

    # bounding boxes
    for obj in xml.iter('object'):
        bbox = obj.find('bndbox')
        
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text) - 1
            bndbox.append(cur_pt)
        box_all += [bndbox]
    return box_all, width

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

def _originvis(name, bbox):
    img = cv2.imread(os.path.join(src_traindir, name))
    for box in bbox:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)
    plt.subplot(1, 1, 1); plt.imshow(img[:, :, [2,1,0]])
    plt.show()
    cv2.waitKey(0)

    
if __name__ == '__main__':
    train_list = glob.glob(src_valdir + '/images/*.jpg')

    temp = []
    for img_path in train_list:
        img = cv2.imread(img_path)
        temp.append(img.shape[1] / img.shape[0])
        # boxes = get_box_label(img_path)
        # _boxvis(img, boxes)
    print(np.unique(np.array(temp)))
    