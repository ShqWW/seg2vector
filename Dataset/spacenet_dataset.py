from .base_train_dataset import BaseTrSet
from .base_test_dataset import BaseTsSet
import os
import json

class SpacenetTrSet(BaseTrSet):
    def __init__(self, cfg=None, transforms=None):
        self.dataset_dir = os.path.join(cfg.dataset_dir, 'RGB_1.0_meter')
        self.preprocess_dir = cfg.preprocess_dir
        self.mask_dir = os.path.join(cfg.preprocess_dir, 'masks')
        self.graph_dir = os.path.join(cfg.preprocess_dir, 'graphs')

        with open(os.path.join(self.dataset_dir, 'dataset.json'), 'r', encoding='utf-8') as file:
            data_split_json = json.load(file)
        exclude_list = ['AOI_3_Paris_39', 'AOI_4_Shanghai_1988', 'AOI_4_Shanghai_1684', 'AOI_3_Paris_448', 'AOI_4_Shanghai_1807', 'AOI_4_Shanghai_1894', 'AOI_4_Shanghai_995', 'AOI_4_Shanghai_1614', 'AOI_3_Paris_299'] #data with no road
        meta_name_list_train = [i for i in data_split_json['train'] if i not in exclude_list]
        meta_name_list_val = [i for i in data_split_json['validation'] if i not in exclude_list]
        self.meta_name_list = meta_name_list_train + meta_name_list_val
        self.anno_name_list = [meta_name + '__gt_graph.p' for meta_name in self.meta_name_list]
        self.img_name_list = [meta_name + '__rgb.png' for meta_name in self.meta_name_list]

        super().__init__(cfg=cfg, transforms=transforms)

class SpacenetTsSet(BaseTsSet):
    def __init__(self, cfg=None, transforms=None):
        self.dataset_dir = os.path.join(cfg.dataset_dir, 'RGB_1.0_meter')

        with open(os.path.join(self.dataset_dir, 'dataset.json'), 'r', encoding='utf-8') as file:
            data_split_json = json.load(file)

        meta_name_list = [i for i in data_split_json['test']]
        self.img_name_list = [meta_name + '__rgb.png' for meta_name in meta_name_list]

        self.is_debug = cfg.is_debug
        if self.is_debug:
            meta_name_list = [i for i in data_split_json['train']]
            self.img_name_list = [meta_name + '__rgb.png' for meta_name in meta_name_list]

        super().__init__(cfg=cfg, transforms=transforms)

