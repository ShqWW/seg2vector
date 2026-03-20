from .base_train_dataset import BaseTrSet
from .base_test_dataset import BaseTsSet
import os
import json
class GlobalscaleTrSet(BaseTrSet):
    def __init__(self, cfg=None, transforms=None):
        self.dataset_dir = os.path.join(cfg.dataset_dir, 'train')
        all_data = os.listdir(self.dataset_dir)
        self.preprocess_dir = cfg.preprocess_dir
        self.mask_dir = os.path.join(cfg.preprocess_dir, 'masks')
        self.graph_dir = os.path.join(cfg.preprocess_dir, 'graphs')

        # train_no_list = [str(x) for x in range(0, 2375)] # trian only
        train_no_list = [str(x) for x in range(0, 2714)] # trian + val
        self.meta_name_list = [f'region_{i}' for i in train_no_list]
        self.anno_name_list = [f'region_{i}_refine_gt_graph.p' for i in train_no_list]
        self.img_name_list = [f'region_{i}_sat.png' for i in train_no_list]
        super().__init__(cfg=cfg, transforms=transforms)


class GlobalscaleTsSet(BaseTsSet):
    def __init__(self, cfg=None, transforms=None):
        self.in_domian_test = cfg.in_domain_test
        if self.in_domian_test:
            self.dataset_dir = os.path.join(cfg.dataset_dir, 'in-domain-test')
            test_no_list = [str(x) for x in range(624)]
        else:
            self.dataset_dir = os.path.join(cfg.dataset_dir, 'out_of_domain')
            test_no_list = [str(x) for x in range(130)]

        self.img_name_list = [f'region_{i}_sat.png' for i in test_no_list]

        self.is_debug = cfg.is_debug
        if self.is_debug:
            self.dataset_dir = os.path.join(cfg.dataset_dir, 'train')
            test_no_list = [str(x) for x in range(2714)]
            self.img_name_list = [f'region_{i}_sat.png' for i in test_no_list]

        super().__init__(cfg=cfg, transforms=transforms)

