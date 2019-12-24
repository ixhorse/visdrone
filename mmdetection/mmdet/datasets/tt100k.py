from .xml_style import XMLDataset
from .coco import CocoDataset
from pycocotools.coco import COCO

class TT100KDataset(XMLDataset):

    CLASSES = (
            'p11', 'pl5', 'pne', 'il60', 'pl80', 'pl100', 'il80', 'po', 'w55',
            'pl40', 'pn', 'pm55', 'w32', 'pl20', 'p27', 'p26', 'p12', 'i5',
            'pl120', 'pl60', 'pl30', 'pl70', 'pl50', 'ip', 'pg', 'p10', 'io',
            'pr40', 'p5', 'p3', 'i2', 'i4', 'ph4', 'wo', 'pm30', 'ph5', 'p23',
            'pm20', 'w57', 'w13', 'p19', 'w59', 'il100', 'p6', 'ph4.5')

    def __init__(self, **kwargs):
        super(TT100KDataset, self).__init__(**kwargs)

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