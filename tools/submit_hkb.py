import cv2
import os, sys
import glob
import json
import shutil
import argparse
import numpy as np
import utils
import pdb

from datasets import HKB

def parse_args():
    parser = argparse.ArgumentParser(description='HKB submit')
    parser.add_argument('--split', type=str, default='val', help='split')
    args = parser.parse_args()
    return args

def eval(gt, det):
    gt_bbox_num = 0
    det_bbox_num = 0
    tp = 0

    for img in gt.keys():
        gt_bboxes = gt[img]
        gt_bbox_num += len(gt_bboxes)
        if img not in det:
            continue
        det_bbox_num += len(det[img])
        for det_bbox in det[img]:
            det_bbox = [det_bbox[0],
                        det_bbox[1],
                        det_bbox[0] + det_bbox[2],
                        det_bbox[1] + det_bbox[3]]
            if (utils.iou_calc1(gt_bboxes, det_bbox) >= 0.5).any():
                tp += 1

    recall = tp / gt_bbox_num
    precision = tp / det_bbox_num
    print(gt_bbox_num)
    print(det_bbox_num)
    print(tp)
    return 2. / (1. / recall + 1. / precision)

if __name__ == '__main__':
    args = parse_args()

    dataset = HKB()
    region_vocdir = dataset.region_voc_dir
    region_imagedir = region_vocdir + '/JpegImages'
    detect_vocdir = dataset.detect_voc_dir
    detect_anno_dir = detect_vocdir + '/Annotations'

    loc_file = detect_anno_dir + '/%s_chip.json' % args.split
    result_file = '../mmdetection/visdrone/results_region_hkb.json'

    with open(result_file, 'r') as f:
        results = json.load(f)
    with open(loc_file, 'r') as f:
        chip_loc = json.load(f)

    # merge detections
    detections = dict()
    for det in results:
        img_id = det['image_id']
        cls_id = det['category_id'] + 1
        bbox = det['bbox']
        score = det['score']

        loc = chip_loc[img_id[:-4]]['loc']
        img_name = '_'.join(img_id.split('_')[:-1]) + '.jpg'
        bbox = [bbox[0] + loc[0], bbox[1] + loc[1], bbox[2], bbox[3]]
        
        if img_name in detections:
            detections[img_name].append(bbox + [score, cls_id])
        else:
            detections[img_name] = [bbox + [score, cls_id]]

    for img_name, det in detections.items():
        det = utils.nms(det, score_threshold=0.4)
        detections[img_name] = det

    # save detections
    with open('results_full_hkb.json', 'w') as f:
        json.dump(detections, f, cls=utils.MyEncoder)
        print('write results json.')

    # read gt
    gt = dict()
    for img_path in dataset.get_imglist(split='val'):
        gt[os.path.basename(img_path)] = dataset.get_gtbox(img_path)[0]

    # eval
    print(eval(gt, detections))