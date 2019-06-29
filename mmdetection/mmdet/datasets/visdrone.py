from .xml_style import XMLDataset
from .coco import CocoDataset
from pycocotools.coco import COCO

class VisDroneDataset(XMLDataset):

    CLASSES = ('1', '2', '3', '4', '5', '6', '7',
               '8', '9', '10')

    def __init__(self, **kwargs):
        super(VisDroneDataset, self).__init__(**kwargs)

# class VisDroneDataset(CocoDataset):

#     CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7',
#                '8', '9')

#     def load_annotations(self, ann_file):
#         self.coco = COCO(ann_file)
#         self.cat_ids = self.coco.getCatIds()
#         self.cat2label = {
#             cat_id: i
#             for i, cat_id in enumerate(self.cat_ids)
#         }
#         self.img_ids = self.coco.getImgIds()
#         img_infos = []
#         for i in self.img_ids:
#             info = self.coco.loadImgs([i])[0]
#             info['filename'] = info['file_name']
#             img_infos.append(info)
#         return img_infos