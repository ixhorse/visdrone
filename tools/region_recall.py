import os, sys
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from path import Path
import utils
import pdb

dbroot = Path.db_root_dir()

src_valdir = dbroot + '/VisDrone2019-DET-val'
mask_path = '../pytorch-deeplab-xception/run/mask'

def main():
    label_object = []
    detect_object = []
    mask_object = []
    undetected_img = []
    pixel_num = []
    for raw_file in tqdm(glob(mask_path + '/*.png')):
        img_name = os.path.basename(raw_file)
        imgid = os.path.splitext(img_name)[0]
        label_file = os.path.join(src_valdir, 'annotations', 'imgid'+'.txt')
        image_file = os.path.join(src_valdir, 'images', imgid + '.jpg')
        
        img = cv2.imread(image_file)
        height, width = img.shape[:2]
        mask_img = cv2.imread(raw_file, cv2.IMREAD_GRAYSCALE)
        mask_h, mask_w = mask_img.shape[:2]
        
        pixel_num.append(np.sum(mask_img))

        label_box, _ = utils.get_box_label(image_file)
        region_box, contours = utils.generate_box_from_mask(mask_img)
        region_box = utils.region_postprocess(region_box, contours, (mask_w, mask_h))
        region_box = utils.resize_box(region_box, (mask_w, mask_h), (width, height))
        region_box = utils.generate_crop_region(region_box, (width, height))

        count = 0
        for box1 in label_box:
            for box2 in region_box:
                if utils.overlap(box2, box1):
                    count += 1
                    break

        label_object.append(len(label_box))
        detect_object.append(count)
        mask_object.append(len(region_box))
        if len(label_box) != count:
            undetected_img.append(imgid)

    print('recall: %f' % (np.sum(detect_object) / np.sum(label_object)))
    # print('cost avg: %f, std: %f' % (np.mean(pixel_num), np.std(pixel_num)))
    print('detect box avg: %f' %(np.mean(mask_object)))
    # print(undetected_img)

if __name__ == '__main__':
    main()