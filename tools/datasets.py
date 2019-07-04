import os, sys
import glob
import numpy as np

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