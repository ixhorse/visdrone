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

    loc_file = detect_anno_dir + '/%s_chip.json' % args.split
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
    with open('results_full_hkb.json', 'w') as f:
        json.dump(detections, f, cls=utils.MyEncoder)
        print('write results json.')

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
            miss_img = ['00010345.jpg', '10000799.jpg', '10004270.jpg', '00013205.jpg', '00011375.jpg', '00000048.jpg', '00010146.jpg', '00010317.jpg', '00011426.jpg', '00010353.jpg', '00010523.jpg', '00011818.jpg', '00011987.jpg', '00011070.jpg', '00010977.jpg', '00011963.jpg', '00011767.jpg', '00011982.jpg', '00011593.jpg', '00011256.jpg', '00012006.jpg', '00011491.jpg', '00012930.jpg', '00013086.jpg', '00010530.jpg', '00010017.jpg', '00011602.jpg', '00011485.jpg', '00012637.jpg', '00013098.jpg', '00011812.jpg', '00011991.jpg', '00011776.jpg', '00012445.jpg', '00012683.jpg', '00011204.jpg', '00011265.jpg', '00010722.jpg', '00011928.jpg', '00011577.jpg', '00010117.jpg', '00011280.jpg', '10000169.jpg', '00012667.jpg', '00012046.jpg', '00010038.jpg', '00011824.jpg', '00011766.jpg', '00010171.jpg', '00012836.jpg', '00012374.jpg', '00012665.jpg']
            for i, img_name in enumerate(miss_img):
                sys.stdout.write('\rprocess: {:d}/{:d}'.format(i + 1, len(miss_img)))
                sys.stdout.flush()

                img_path = os.path.join(src_imagedir, img_name)
                img = cv2.imread(img_path)
                for det in detections[img_name]:
                    bbox = det[:4]
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 3)
                for gt_box in gt[img_name]:
                    bbox = gt_box
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                for loc in regions[img_name]:
                    cv2.rectangle(img, (int(loc[0]), int(loc[1])), (int(loc[2]), int(loc[3])), (255, 0, 0), 2)
                cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), img)