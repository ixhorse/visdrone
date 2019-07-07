import os, sys
import glob
import numpy as np
import xml.etree.ElementTree as ET

class Dataset(object):
    def __init__(self):
        # init dataset path
        self.user_home = os.path.expanduser('~')
        pass

    def _init_path(self, dbroot):
        self.region_voc_dir = dbroot + '/region_voc'
        self.detect_voc_dir = dbroot + '/detect_voc'
        self.detect_coco_dir = dbroot + '/detect_coco'

    def get_imglist(self, split='train'):
        """
        return list of all image paths
        """
        pass

    def get_gtbox(self, img_name):
        """
        Return gt bounding boxes of original image
        """
        pass

class VisDrone(Dataset):
    def __init__(self):
        super(VisDrone, self).__init__()

        user_home = self.user_home
        self.db_root = os.path.join(user_home, 'data/visdrone2019')
        self.src_traindir = self.db_root + '/VisDrone2019-DET-train'
        self.src_valdir = self.db_root + '/VisDrone2019-DET-val'
        self.src_testdir = self.db_root + '/VisDrone2019-DET-test-challenge'
        self._init_path(self.db_root)

    def get_imglist(self, split='train'):
        if split == 'train':
            return glob.glob(self.src_traindir + '/images/*.jpg')
        elif split == 'val':
            return glob.glob(self.src_valdir + '/images/*.jpg')
        elif split == 'test':
            return glob.glob(self.src_testdir + '/images/*.jpg')
        else:
            raise('error')

    def get_gtbox(self, img_path):
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


class HKB(Dataset):
    def __init__(self):
        super(HKB, self).__init__()

        user_home = self.user_home
        self.db_root = os.path.join(user_home, 'data/HKB')
        self.src_imagedir = self.db_root + '/JPEGImages'
        self.src_annodir = self.db_root + '/Annotations'
        self._trainval_split()
        self._init_path(self.db_root)

    def _trainval_split(self):
        np.random.seed(666)
        imglist = os.listdir(self.src_imagedir)
        np.random.shuffle(imglist)
        self.train_list = imglist[:-400]
        self.val_list = imglist[-400:]

    def get_imglist(self, split='train'):
        if split == 'train':
            return [os.path.join(self.src_imagedir, x) for x in self.train_list]
        elif split == 'val':
            return [os.path.join(self.src_imagedir, x) for x in self.val_list]
        elif split == 'test':
            return [os.path.join(self.src_imagedir, x) for x in self.val_list]
        else:
            raise NotImplementedError

    def get_annolist(self, split='trainval'):
        if split == 'train':
            return [os.path.join(self.src_annodir, x[:-4]+'.xml') for x in self.train_list]
        if split == 'val':
            return [os.path.join(self.src_annodir, x[:-4]+'.xml') for x in self.val_list]
        else:
            raise NotImplementedError

    def get_gtbox(self, img_path, return_size=False):
        anno_path = img_path.replace('JPEGImages', 'Annotations')
        anno_path = anno_path.replace('jpg', 'xml')
        xml = ET.parse(anno_path).getroot()
        box_all = []
        label_all = []
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        # bounding boxes
        for obj in xml.iter('object'):
            label_all.append(1)
            bbox = obj.find('bndbox')
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text)
                bndbox.append(cur_pt)
            box_all += [bndbox]
        
        if not return_size:
            return box_all, label_all
        else:
            size = xml.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            return (width, height), box_all, label_all

def get_dataset(dataset_name):
    if dataset_name == 'VisDrone':
        return VisDrone()
    elif dataset_name == 'HKB':
        return HKB()
    else:
        raise NotImplementedError

if __name__ == '__main__':
    dataset = HKB()
    train_list = dataset.get_imglist('train')
    print(len(train_list))
    all_size = []
    for img_path in train_list:
        size, _, labels = dataset.get_gtbox(img_path, return_size=True)
        all_size.append(str(size))

    print(np.unique(np.array(all_size)))