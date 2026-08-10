from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS

@DATASETS.register_module()
class SODA_A_HBBDataset(CocoDataset):
    METAINFO = {
        'classes':
        ('airplane', 'helicopter', 'small-vehicle', 'large-vehicle',
        'ship', 'container', 'storage-tank', 'swimming-pool',
        'windmill'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':[
            (255, 0, 0),    # 红色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 品红
            (0, 255, 255),  # 青色
            (128, 0, 128),  # 紫色
            (128, 128, 0),  # 橄榄色
            (0, 128, 128)   # 青绿色
        ]
    }
    
    def __init__(self, *args, **kwargs):
        """
        初始化 SODA_A_HBBDataset。
        """
        super().__init__(*args, **kwargs)