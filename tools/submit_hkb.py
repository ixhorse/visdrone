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
    parser.add_argument('--eval', action='store_true', help='F1')
    parser.add_argument('--save_det', action='store_true', help='save detection results')
    args = parser.parse_args()
    return args

def eval(gt, det):
    gt_bbox_num = 0
    det_bbox_num = 0
    tp = 0

    miss_img = []

    for img in gt.keys():
        gt_bboxes = gt[img]
        img_tp = 0

        gt_bbox_num += len(gt_bboxes)
        if img not in det:
            continue
        det_bbox_num += len(det[img])

        for det_bbox in det[img]:
            if (utils.iou_calc1(gt_bboxes, det_bbox[:4]) >= 0.5).any():
                img_tp += 1
        
        tp += img_tp
        if(img_tp != len(det[img]) or img_tp != len(gt[img])):
            miss_img.append(img)

    recall = tp / gt_bbox_num
    precision = tp / det_bbox_num
    print('gt_bbox_num = %d' % gt_bbox_num)
    print('det_bbox_num = %d' % det_bbox_num)
    print('tp = %d' % tp)
    print('F1 = %f' % (2. / (1. / recall + 1. / precision)))

    return sorted(miss_img)

if __name__ == '__main__':
    args = parse_args()

    dataset = HKB()
    src_imagedir = dataset.src_imagedir
    region_vocdir = dataset.region_voc_dir
    region_imagedir = region_vocdir + '/JpegImages'
    detect_vocdir = dataset.detect_voc_dir
    detect_anno_dir = detect_vocdir + '/Annotations'

    loc_file = detect_anno_dir + '/test_chip.json'
    result_file = '../mmdetection/visdrone/results_region_hkb.json'

    with open(result_file, 'r') as f:
        results = json.load(f)
    with open(loc_file, 'r') as f:
        chip_loc = json.load(f)

    # merge detections
    detections = dict()
    regions = dict()
    for det in results:
        img_id = det['image_id']
        cls_id = det['category_id'] + 1
        bbox = det['bbox'] #[xmin, ymin, width, height]
        score = det['score']

        loc = chip_loc[img_id[:-4]]['loc']
        img_name = '_'.join(img_id.split('_')[:-1]) + '.jpg'
        bbox = [bbox[0] + loc[0],
                bbox[1] + loc[1],
                bbox[0] + loc[0] + bbox[2],
                bbox[1] + loc[1] + bbox[3]] #[xmin, ymin, xmax, ymax]
        
        if img_name in detections:
            detections[img_name].append(bbox + [score, cls_id])
        else:
            detections[img_name] = [bbox + [score, cls_id]]
        if img_name in regions:
            regions[img_name].append(loc)
        else:
            regions[img_name] = [loc]

    # nms
    for img_name, det in detections.items():
        det = utils.nms(det, score_threshold=0.4, overlap_threshold=0.92)
        detections[img_name] = det

    # save detections
    img_paths = dataset.get_imglist(split='test')
    output_dir = 'hkb_det_txt'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        det = detections[img_name] if img_name in detections else []
        txt_name = img_name[:-4] + '.txt'
        with open(os.path.join(output_dir, txt_name), 'w') as f:
            for bbox in det:
                bbox = [str(int(x)) for x in bbox[0:4]] + ['Vehicle']
                f.write(','.join(bbox) + '\n')

    if args.eval:
        # read gt
        gt = dict()
        for img_path in dataset.get_imglist(split='val'):
            gt[os.path.basename(img_path)] = dataset.get_gtbox(img_path)[0]

        # eval
        miss_img = eval(gt, detections)
    
    if args.save_det:
        output_dir = 'hkb_det_save'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        # draw boxes
        for i, img_path in enumerate(img_paths):
            sys.stdout.write('\rprocess: {:d}/{:d}'.format(i + 1, len(img_paths)))
            sys.stdout.flush()

            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)
            
            if not img_name in detections:
                continue
            
            for det in detections[img_name]:
                bbox = det[:4]
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 3)
            if args.eval:
                for gt_box in gt[img_name]:
                    bbox = gt_box
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            for loc in regions[img_name]:
                cv2.rectangle(img, (int(loc[0]), int(loc[1])), (int(loc[2]), int(loc[3])), (255, 0, 0), 2)
            cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), img)