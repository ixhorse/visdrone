import cv2
import os, sys
import glob
import json
import shutil
import numpy as np
from path import Path
import utils
import pdb

dbroot = Path.db_root_dir()

src_traindir = dbroot + '/VisDrone2019-DET-train'
src_valdir = dbroot + '/VisDrone2019-DET-val'
src_testdir = dbroot + '/VisDrone2019-DET-test-challenge'

voc_datadir = dbroot + '/detect_voc'
voc_anno_dir = voc_datadir + '/Annotations'

coco_datadir = dbroot + '/detect_coco'
coco_image_dir = coco_datadir + '/images'
coco_anno_dir = coco_datadir + '/annotations'

val_anno = coco_anno_dir + '/instances_val.json'
result_file =  '../mmdetection/visdrone/results.pkl.bbox.json'
loc_file = voc_anno_dir + '/val_chip.json'

if __name__ == '__main__':
    with open(val_anno, 'r') as f:
        annos = json.load(f)
    with open(result_file, 'r') as f:
        results = json.load(f)
    with open(loc_file, 'r') as f:
        chip_loc = json.load(f)

    img_id2name = dict()
    for img in annos['images']:
        img_id2name[img['id']] = img['file_name']

    detecions = dict()
    for det in results:
        img_id = det['image_id']
        cls_id = det['category_id'] + 1
        bbox = det['bbox']
        score = det['score']
        loc = chip_loc[img_id2name[img_id]]
        bbox = [bbox[0] + loc[0], bbox[1] + loc[1], bbox[2], bbox[3]]
        img_name = '_'.join(img_id2name[img_id].split('_')[:-1]) + '.jpg'
        if img_name in detecions:
            detecions[img_name].append(bbox + [score, cls_id])
        else:
            detecions[img_name] = [bbox + [score, cls_id]]

    output_dir = 'DET_results-val'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    for img_name, det in detecions.items():
        det = utils.nms(det)
        txt_name = img_name[:-4] + '.txt'
        with open(os.path.join(output_dir, txt_name), 'w') as f:
            for bbox in det:
                bbox = [str(x) for x in (list(bbox[0:5]) + [int(bbox[5])] + [-1, -1])]
                f.write(','.join(bbox) + '\n')