from .base_train_dataset import BaseTrSet
from .base_test_dataset import BaseTsSet
import os
import json

class CityscaleTrSet(BaseTrSet):
    def __init__(self, cfg=None, transforms=None):
        self.dataset_dir = os.path.join(cfg.dataset_dir, '20cities')
        self.preprocess_dir = cfg.preprocess_dir
        self.mask_dir = os.path.join(cfg.preprocess_dir, 'masks')
        self.graph_dir = os.path.join(cfg.preprocess_dir, 'graphs')

        with open(os.path.join(cfg.dataset_dir, 'data_split.json'), 'r', encoding='utf-8') as file:
            data_split_json = json.load(file)
        train_no_list = [str(i) for i in data_split_json['train'] + data_split_json['valid']]
        self.meta_name_list = [f'region_{i}' for i in train_no_list]
        self.anno_name_list = [f'region_{i}_refine_gt_graph.p' for i in train_no_list]
        self.img_name_list = [f'region_{i}_sat.png' for i in train_no_list]
        
        super().__init__(cfg=cfg, transforms=transforms)

class CityscaleTsSet(BaseTsSet):
    def __init__(self, cfg=None, transforms=None):
        self.dataset_dir = os.path.join(cfg.dataset_dir, '20cities')

        with open(os.path.join(cfg.dataset_dir, 'data_split.json'), 'r', encoding='utf-8') as file:
            data_split_json = json.load(file)

        test_no_list = [str(i) for i in data_split_json['test']]
        self.img_name_list = [f'region_{i}_sat.png' for i in test_no_list]

        self.is_debug = cfg.is_debug
        if self.is_debug:
            test_no_list = [str(i) for i in data_split_json['train']]
            self.img_name_list = [f'region_{i}_sat.png' for i in test_no_list]

        super().__init__(cfg=cfg, transforms=transforms)

