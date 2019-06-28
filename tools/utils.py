import cv2
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import pdb

def region_cluster(regions, mask_shape):
    """
    层次聚类
    """
    regions = np.array(regions)
    centers = (regions[:, [2, 3]] + regions[:, [0, 1]]) / 2.0

    model = AgglomerativeClustering(
                n_clusters=None,
                linkage='complete',
                distance_threshold=min(mask_shape) * 0.4,
                compute_full_tree=True)

    labels = model.fit_predict(centers)

    cluster_regions = []
    for idx in np.unique(labels):
        boxes = regions[labels == idx]
        new_box = [min(boxes[:, 0]), min(boxes[:, 1]),
                   max(boxes[:, 2]), max(boxes[:, 3])]
        cluster_regions.append(new_box)
    
    return cluster_regions

def region_split(regions, mask_shape):
    alpha = 1
    new_regions = []
    for box in regions:
        width, height = box[2] - box[0], box[3] - box[1]
        if width / height > 1.5:
            mid = int(box[0] + width / 2.0)
            new_regions.append([box[0], box[1], mid + alpha, box[3]])
            new_regions.append([mid - alpha, box[1], box[2], box[3]])
        elif height / width > 1.5:
            mid = int(box[1] + height / 2.0)
            new_regions.append([box[0], box[1], box[2], mid + alpha])
            new_regions.append([box[0], mid - alpha, box[2], box[3]])
        else:
            new_regions.append(box)
    return new_regions

def region_morphology(contours, mask_shape):
    mask_w, mask_h = mask_shape
    binary = np.zeros((mask_h, mask_w)).astype(np.uint8)
    cv2.drawContours(binary, contours, -1, 1, cv2.FILLED)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel) # 开操作
    region_box, _ = generate_box_from_mask(binary)
    regions = []
    for box in region_box:
        box = [max(0, box[0]-2), max(0, box[1]-2),
               min(mask_w, box[2]+2), min(mask_h, box[3]+2)]
        regions.append(box)
    return regions

def region_postprocess(regions, contours, mask_shape):
    mask_w, mask_h = mask_shape

    # 1. get big contours
    small_regions = []
    big_contours = []
    for i, box in enumerate(regions):
        w, h = box[2] - box[0], box[3] - box[1]
        if w > mask_w / 2 or h > mask_h / 2:
            big_contours.append(contours[i])
        else:
            small_regions.append(box)
            
    # 2. image open
    regions = region_morphology(big_contours, mask_shape) + small_regions

    # 3. process small regions and big regions
    small_regions = []
    big_regions = []
    for box in regions:
        w, h = box[2] - box[0], box[3] - box[1]
        if max(w, h) > 12:
            big_regions.append(box)
        else:
            small_regions.append(box)
    if len(big_regions) > 0:
        big_regions = region_split(big_regions, mask_shape)
    if len(small_regions) > 1:
        small_regions = region_cluster(small_regions, mask_shape)
    
    regions = np.array(small_regions + big_regions)

    return regions


def generate_crop_region(regions, img_size):
    """
    generate final regions
    enlarge regions < 300
    """
    width, height = img_size
    final_regions = []
    for box in regions:
        box_w, box_h = box[2] - box[0], box[3] - box[1]
        center_x, center_y = box[0] + box_w / 2.0, box[1] + box_h / 2.0
        if box_w < 300 and box_h < 300:
            crop = True
            crop_size = 300 / 2
        elif box_w / box_h > 1.3 or box_h / box_w > 1.3:
            crop = True
            crop_size = max(box_w, box_h) / 2
        else:
            crop = False

        if crop:
            center_x = crop_size if center_x < crop_size else center_x
            center_y = crop_size if center_y < crop_size else center_y
            center_x = width - crop_size - 1 if center_x > width - crop_size - 1 else center_x
            center_y = height - crop_size - 1 if center_y > height - crop_size - 1 else center_y
            
            new_box = [center_x - crop_size if center_x - crop_size > 0 else 0,
                    center_y - crop_size if center_y - crop_size > 0 else 0,
                    center_x + crop_size if center_x + crop_size < width else width-1,
                    center_y + crop_size if center_y + crop_size < height else height-1]
            for x in new_box:
                if x < 0:
                    pdb.set_trace()
            final_regions.append([int(x) for x in new_box])
        else:
            final_regions.append([int(x) for x in box])

    regions = np.array(final_regions)
    while(1):
        idx = np.zeros((len(regions)))
        for i in range(len(regions)):
            for j in range(len(regions)):
                if i == j or idx[i] == 1 or idx[j] == 1:
                    continue
                box1, box2 = regions[i], regions[j]
                if overlap(regions[i], regions[j], 0.8):
                    regions[i][0] = min(box1[0], box2[0])
                    regions[i][1] = min(box1[1], box2[1])
                    regions[i][2] = max(box1[2], box2[2])
                    regions[i][3] = max(box1[3], box2[3])
                    idx[j] = 1
        if sum(idx) == 0:
            break
        regions = regions[idx == 0]
    return regions


def generate_box_from_mask(mask):
    """
    Args:
        mask: 0/1 array
    """
    regions = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        regions.append([x, y, x+w, y+h])

    # delete inner box
    new_contours = []
    regions = np.array(regions)
    idx = np.zeros((len(regions)))
    for i in range(len(regions)):
        for j in range(len(regions)):
            if i == j or idx[i] == 1 or idx[j] == 1:
                continue
            box1, box2 = regions[i], regions[j]
            if overlap(box1, box2, 0.9):
            # if overlap(box1, box2, 0.6) or \
            #     (box2[2] - box2[0] < 2 and overlap(box1, box2, 0.1)):
                # regions[i][0] = min(box1[0], box2[0])
                # regions[i][1] = min(box1[1], box2[1])
                # regions[i][2] = max(box1[2], box2[2])
                # regions[i][3] = max(box1[3], box2[3])
                idx[j] = 1
    regions = regions[idx == 0]

    for i, id in enumerate(idx):
        if id == 0:
            new_contours.append(contours[i])
    return regions, new_contours


def enlarge_box(mask_box, image_size, ratio=2):
    """
    Args:
        mask_box: list of box
        image_size: (width, height)
        ratio: int
    """
    new_mask_box = []
    for box in mask_box:
        w = box[2] - box[0]
        h = box[3] - box[1]
        center_x = w / 2 + box[0]
        center_y = h / 2 + box[1]
        w = w * ratio / 2
        h = h * ratio / 2
        new_box = [center_x-w if center_x-w > 0 else 0,
                    center_y-h if center_y-h > 0 else 0,
                    center_x+w if center_x+w < image_size[0] else image_size[0]-1,
                    center_y+h if center_y+h < image_size[1] else image_size[1]-1]
        new_box = [int(x) for x in new_box]
        new_mask_box.append(new_box)
    return new_mask_box


def resize_box(box, original_size, dest_size):
    """
    Args:
        box: array, [xmin, ymin, xmax, ymax]
        original_size: (width, height)
        dest_size: (width, height)
    """
    h_ratio = 1.0 * dest_size[1] / original_size[1]
    w_ratio = 1.0 * dest_size[0] / original_size[0]
    box = np.array(box) * np.array([w_ratio, h_ratio, w_ratio, h_ratio])
    return list(box.astype(np.int32))


def overlap(box1, box2, thresh = 0.75):
    """ (box1 \cup box2) / box2
    Args:
        box1: [xmin, ymin, xmax, ymax]
        box2: [xmin, ymin, xmax, ymax]
    """
    matric = np.array([box1, box2])
    u_xmin = max(matric[:,0])
    u_ymin = max(matric[:,1])
    u_xmax = min(matric[:,2])
    u_ymax = min(matric[:,3])
    u_w = u_xmax - u_xmin
    u_h = u_ymax - u_ymin
    if u_w <= 0 or u_h <= 0:
        return False
    u_area = u_w * u_h
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    if u_area / box2_area < thresh:
        return False
    else:
        return True

def get_box_label(img_path):
    anno_path = img_path.replace('images', 'annotations')
    anno_path = anno_path.replace('jpg', 'txt')
    with open(anno_path, 'r') as f:
        data = [x.strip().split(',')[:8] for x in f.readlines()]
        annos = np.array(data)

    boxes = annos[annos[:, 4] == '1'][:, :6].astype(np.int32)
    y = np.zeros((len(boxes), 4)).astype(np.int32)
    y[:, 0] = boxes[:, 0]
    y[:, 1] = boxes[:, 1]
    y[:, 2] = boxes[:, 0] + boxes[:, 2]
    y[:, 3] = boxes[:, 1] + boxes[:, 3]

    labels = boxes[:, 5]
    
    return y, labels