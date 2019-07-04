"""
generate detect dataset in VOC format
"""
import cv2
import random
import os, sys
import glob
import json
import numpy as np
from tqdm import tqdm
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from datasets import VisDrone
import utils

import pdb
import traceback

def parse_args():
    parser = argparse.ArgumentParser(description="convert to voc dataset")
    parser.add_argument('mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='for train or test')
    args = parser.parse_args()
    return args

def generate_region_gt(img_size, region_box, gt_boxes, labels):
    chip_list = []
    for box in region_box:
        chip_list.append(np.array(box))
    
    # chip gt
    chip_gt_list = []
    chip_label_list = []
    for chip in chip_list:
        chip_gt = []
        chip_label = []

        for i, box in enumerate(gt_boxes):
            if labels[i] == 0 or labels[i] == 11:
                continue
            if utils.overlap(chip, box, 0.75):
                box = [max(box[0], chip[0]), max(box[1], chip[1]), 
                       min(box[2], chip[2]), min(box[3], chip[3])]
                new_box = [box[0] - chip[0], box[1] - chip[1],
                           box[2] - chip[0], box[3] - chip[1]]

                chip_gt.append(np.array(new_box))
                chip_label.append(labels[i])

        chip_gt_list.append(chip_gt)
        chip_label_list.append(chip_label)
    
    return chip_list, chip_gt_list, chip_label_list


def make_xml(chip, box_list, label_list, image_name, tsize):

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name

    node_object_num = SubElement(node_root, 'object_num')
    node_object_num.text = str(len(box_list))

    node_location = SubElement(node_root, 'location')
    node_loc_xmin = SubElement(node_location, 'xmin')
    node_loc_xmin.text = str(int(chip[0]) + 1)
    node_loc_ymin = SubElement(node_location, 'ymin')
    node_loc_ymin.text = str(int(chip[1]) + 1)
    node_loc_xmax = SubElement(node_location, 'xmax')
    node_loc_xmax.text = str(int(chip[2]) + 1)
    node_loc_ymax = SubElement(node_location, 'ymax')
    node_loc_ymax.text = str(int(chip[3]) + 1)

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(tsize[0])
    node_height = SubElement(node_size, 'height')
    node_height.text = str(tsize[1])
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for i in range(len(box_list)):  
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = str(label_list[i])
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        # voc dataset is 1-based
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(box_list[i][0]) + 1)
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(box_list[i][1]) + 1)
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(box_list[i][2] + 1))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(box_list[i][3] + 1))

    xml = tostring(node_root, encoding='utf-8')
    dom = parseString(xml)
    # print(xml)
    return dom


def write_chip_and_anno(image, imgid, 
    chip_list, chip_gt_list, chip_label_list):
    """write chips of one image to disk and make xml annotations
    """
    assert len(chip_gt_list) > 0
    chip_loc = dict()
    for i, chip in enumerate(chip_list):
        img_name = '%s_%d.jpg' % (imgid, i)
        xml_name = '%s_%d.xml' % (imgid, i)
        chip_loc[img_name] = [int(x) for x in chip]
        chip_size = (chip[2] - chip[0], chip[3] - chip[1])
        
        chip_img = image[chip[1]:chip[3], chip[0]:chip[2], :].copy()
        assert len(chip_img.shape) == 3

        bbox = []
        for gt in chip_gt_list[i]:
            bbox.append(gt)
        bbox = np.array(bbox, dtype=np.int)

        dom = make_xml(chip, bbox, chip_label_list[i], img_name, chip_size)

        cv2.imwrite(os.path.join(image_dir, img_name), chip_img)
        with open(os.path.join(anno_dir, xml_name), 'w') as f:
            f.write(dom.toprettyxml(indent='\t', encoding='utf-8').decode('utf-8'))
        
    return chip_loc


def generate_imgset(img_list, imgset):
    with open(os.path.join(list_dir, imgset+'.txt'), 'w') as f:
        f.writelines([x + '\n' for x in img_list])
    print('\n%d images in %s set.' % (len(img_list), imgset))


def _worker(img_path, dataset):
    try:
        imgid = os.path.basename(img_path)[:-4]
        image = cv2.imread(img_path)
        height, width = image.shape[:2]

        mask_path = os.path.join(segmentation_dir, imgid+'_region.png')
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_h, mask_w = mask_img.shape[:2]
        region_box, contours = utils.generate_box_from_mask(mask_img)
        region_box = utils.region_postprocess(region_box, contours, (mask_w, mask_h))
        region_box = utils.resize_box(region_box, (mask_w, mask_h), (width, height))
        region_box = utils.generate_crop_region(region_box, (width, height))

        gt_boxes, labels = dataset.get_gtbox(img_path)

        chip_list, chip_gt_list, chip_label_list = generate_region_gt((width, height), region_box, gt_boxes, labels)
        chip_loc = write_chip_and_anno(image, imgid, chip_list, chip_gt_list, chip_label_list)
        return len(chip_list), chip_loc
        # _progress()
    except Exception:
        traceback.print_exc()
        os._exit(0) 


def main():
    args = parse_args()

    dataset = VisDrone()
    region_dir = dataset.region_voc_dir
    segmentation_dir = region_dir + '/SegmentationClass'

    dest_datadir = dataset.detect_voc_dir
    image_dir = dest_datadir + '/JPEGImages'
    anno_dir = dest_datadir + '/Annotations'
    list_dir = dest_datadir + '/ImageSets/Main'

    if not os.path.exists(dest_datadir):
        os.mkdir(dest_datadir)
        os.mkdir(image_dir)
        os.makedirs(list_dir)
        os.mkdir(anno_dir)
    
    train_list = dataset.get_imglist('train')
    val_list = dataset.get_imglist('val')
    trainval_list = train_list + val_list

    for img_list, imgset in zip([val_list], ['val']):
        chip_ids = []
        chip_loc = dict()
        for i, img_path in enumerate(img_list):
            img_id = os.path.basename(img_list[i])[:-4]
            
            sys.stdout.write('\rsearch: {:d}/{:d} {:s}'
                            .format(i + 1, len(img_list), img_id))
            sys.stdout.flush()

            chiplen, loc = _worker(img_path, dataset)
            for i in range(chiplen):
                chip_ids.append('%s_%s' % (img_id, i))
            chip_loc.update(loc)
        
        generate_imgset(chip_ids, imgset)

        # write chip loc json
        with open(os.path.join(anno_dir, '%s_chip.json'%imgset), 'w') as f:
            json.dump(chip_loc, f)
            print('write loc json.')

if __name__ == '__main__':
    main()