from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS


'''
COCO_CATEGORY_MAP = {
    'car': 1,
    'boat': 2, 
    'tank': 3,
    'truck': 5,
    'boarding_bridge': 6,
    'crane': 7,
    'airplane': 8,
    'lattice_tower': 9
}
'''

@DATASETS.register_module()
class STAR_SUB8KDataset(CocoDataset):
    METAINFO = {
        'classes':
        ('car', 'boat', 'tank','truck','boarding_bridge','crane','airplane','lattice_tower'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [
            (255, 0, 0),      # car - 红色
            (0, 255, 0),      # boat - 绿色
            (0, 0, 255),      # tank - 蓝色
            (255, 255, 0),    # truck - 黄色
            (0, 255, 255),    # boarding_bridge - 青色
            (255, 0, 255),    # crane - 品红
            (255, 165, 0),    # airplane - 橙色
            (128, 0, 128),    # lattice_tower - 紫色
        ]
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)