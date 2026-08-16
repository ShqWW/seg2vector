import torch
from torch.utils.data import Dataset
import random
import numpy as np
import albumentations as A
import os 
import cv2
from utils.graph_utils import *
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool


from functools import partial
from utils.graph_utils import *
from utils.patch_utils import *
from utils.angle_utils import *
from utils.point_utils import *
from .label_preprocess import graph_preprocess, mask_preprocess



# debug import 
# import time
# from pathos import multiprocessing
# from utils.link_utils import *
# from utils.plot_utils import *
# from utils.nms_utils import *


class BaseTrSet(Dataset):
    def __init__(self, cfg=None, transforms=None):
        random.seed(cfg.random_seed)
        self.transforms = transforms
        self.img_h, self.img_w = cfg.img_h, cfg.img_w
        self.patch_h, self.patch_w = cfg.patch_h, cfg.patch_w
        self.patch_num_h, self.patch_num_w = cfg.patch_num_h_training, cfg.patch_num_w_training
        self.num_patch_per_img = self.patch_num_h * self.patch_num_w
        self.margin_size = cfg.margin_size
        self.max_node_degree = cfg.max_node_degree

        self.is_buffer = cfg.is_buffer
        self.is_preprocess = cfg.is_preprocess
        self.is_multiprocess = cfg.is_multiprocess
        self.num_processes = cfg.num_processes
        self.poly_angle_thres = cfg.poly_angle_thres
        self.train_vector = cfg.train_vector

        self.vector_teacher = cfg.VECTOR_TEACHER

        self.is_debug = cfg.is_debug

        if self.is_debug:
            self.meta_name_list = self.meta_name_list[:2]
            self.anno_name_list = self.anno_name_list[:2]
            self.img_name_list = self.img_name_list[:2]


        self.num_train_data = len(self.anno_name_list)


        ################################image augment##################################
        img_transforms = []
        self.aug_names = cfg.train_augments
        for aug in self.aug_names:
            if aug['name'] != 'OneOf':
                img_transforms.append(getattr(A, aug['name'])(**aug['parameters']))
            else:
                img_transforms.append(A.OneOf([getattr(A, aug_['name'])(**aug_['parameters'])
                                      for aug_ in aug['transforms']], p=aug['p']))
        self.train_augments = A.Compose(img_transforms, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),  additional_targets={'mask': 'mask'})
        ####################################################################

        if self.is_preprocess:
            self.preprocess(cfg)
        self.prepare_img()
        self.init_patch_info(self.num_train_data)

        self.label_keyname_list = ['seg_masks']
        if self.train_vector:
            self.label_keyname_list += ['road_sample_points', 'road_target_points', 'road_sample_valids', 'road_angles',
                                        'key_sample_points', 'key_target_points', 'key_sample_valids', 'key_angles', 'key_angle_valids']

        if self.is_debug:
            self.length = self.num_train_data * self.num_patch_per_img * cfg.debug_length_duplicate
            
    def __len__(self):
        if self.is_debug:
            return self.length

        return self.num_train_data * self.num_patch_per_img
        
    def preprocess(self, cfg=None):
        print('Preprocessing dataset ...')
        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.graph_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)
        if self.is_multiprocess:
            with ProcessingPool(processes=self.num_processes) as pool:
                func_graph = partial(graph_preprocess, 
                            label_dir = self.dataset_dir,
                            out_dir = self.graph_dir,
                            cfg=cfg, 
                            )
                func_mask = partial(mask_preprocess, 
                            label_dir = self.dataset_dir,
                            out_dir = self.mask_dir,
                            cfg=cfg, 
                            )
                list(tqdm(pool.imap(func_graph, self.anno_name_list, self.meta_name_list), total=len(self.meta_name_list)))
                list(tqdm(pool.imap(func_mask, self.anno_name_list, self.meta_name_list), total=len(self.meta_name_list)))
        else:
            for i in tqdm(range(len(self.meta_name_list))):
                anno_name, meta_name = self.anno_name_list[i], self.meta_name_list[i]
                graph_preprocess(label_name = anno_name, out_name = meta_name, label_dir = self.dataset_dir, out_dir = self.graph_dir, cfg=cfg)
                mask_preprocess(label_name = anno_name, out_name = meta_name, label_dir = self.dataset_dir, out_dir = self.mask_dir, cfg=cfg)


    def prepare_img(self):
        self.img_path_list = [os.path.join(self.dataset_dir, img_name) for img_name in self.img_name_list]
        self.key_mask_path_list = [os.path.join(self.mask_dir, meta_name + '_keypoint_mask.png') for meta_name in self.meta_name_list]
        self.road_mask_path_list = [os.path.join(self.mask_dir, meta_name + '_road_mask.png') for meta_name in self.meta_name_list]
        # self.overlap_mask_path_list = [os.path.join(self.mask_dir, meta_name + '_overlappoint_mask.png') for meta_name in self.meta_name_list]
        self.graph_path_list = [os.path.join(self.graph_dir, meta_name + '_graph.pkl') for meta_name in self.meta_name_list]


        if self.is_buffer:
            img_buffer = []
            key_mask_buffer = []
            road_mask_buffer = []
            # overlap_mask_buffer = []
            if self.is_multiprocess:
                with ProcessingPool(processes=self.num_processes) as pool:
                    img_buffer = list(tqdm(pool.imap(cv2.imread, self.img_path_list), total=len(self.img_path_list)))
                    func_gray = partial(cv2.imread, 
                                flags = cv2.IMREAD_GRAYSCALE 
                                )
                    key_mask_buffer = list(tqdm(pool.imap(func_gray, self.key_mask_path_list), total=len(self.key_mask_path_list)))
                    road_mask_buffer = list(tqdm(pool.imap(func_gray, self.road_mask_path_list), total=len(self.road_mask_path_list)))
                    # overlap_mask_buffer = list(tqdm(pool.imap(func_gray, self.overlap_mask_path_list), total=len(self.overlap_mask_path_list)))
                self.img_buffer = {self.img_path_list[i]: img_buffer[i] for i in range(len(self.img_path_list))}
                self.key_mask_buffer = {self.key_mask_path_list[i]: key_mask_buffer[i] for i in range(len(self.key_mask_path_list))}
                self.road_mask_buffer = {self.road_mask_path_list[i]: road_mask_buffer[i] for i in range(len(self.road_mask_path_list))}
                # self.overlap_mask_buffer = {self.overlap_mask_path_list[i]: overlap_mask_buffer[i] for i in range(len(self.overlap_mask_path_list))}
                
            else:
                for i in tqdm(range(len(self.img_path_list))):
                    img = cv2.imread(self.img_path_list[i])
                    key_mask = cv2.imread(self.key_mask_path_list[i], cv2.IMREAD_GRAYSCALE)
                    road_mask = cv2.imread(self.road_mask_path_list[i], cv2.IMREAD_GRAYSCALE)
                    # overlap_mask = cv2.imread(self.overlap_mask_path_list[i], cv2.IMREAD_GRAYSCALE)
                    self.img_buffer.append(img)
                    self.key_mask_buffer.append(key_mask)
                    self.road_mask_buffer.append(road_mask)
                    # self.overlap_mask_buffer.append(overlap_mask)

    def read_img(self, index):
        img_path = self.img_path_list[index]
        key_mask_path = self.key_mask_path_list[index]
        road_mask_path = self.road_mask_path_list[index]
        # overlap_mask_path = self.overlap_mask_path_list[index]
        if self.is_buffer:
            img = self.img_buffer[img_path]
            key_mask = self.key_mask_buffer[key_mask_path]
            road_mask = self.road_mask_buffer[road_mask_path]
            # overlap_mask = self.overlap_mask_buffer[overlap_mask_path]
        else:
            img = cv2.imread(img_path)
            key_mask = cv2.imread(key_mask_path, cv2.IMREAD_GRAYSCALE)
            road_mask = cv2.imread(road_mask_path, cv2.IMREAD_GRAYSCALE)
            # overlap_mask = cv2.imread(overlap_mask_path, cv2.IMREAD_GRAYSCALE)

        if img.shape[0] < self.img_h or img.shape[1] < self.img_w:
            real_img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
            real_img[0:img.shape[0], 0:img.shape[1], :] = img
            img = real_img
        # print(img.shape)

        return img, key_mask, road_mask

    def init_patch_info(self, num_img):
        '''
        init the patch info
        the patch info contains the top left of the patch axis within the whole image
        '''
        patch_grid_h = float((self.img_h- (self.patch_h + 2*self.margin_size))/self.patch_num_h)
        patch_grid_w = float((self.img_w- (self.patch_w + 2*self.margin_size))/self.patch_num_w)
        start_h_array = np.linspace(start=self.margin_size, stop=self.img_h- (self.patch_h + self.margin_size), num=self.patch_num_h).astype(np.float32)-patch_grid_h*0.5
        start_w_array = np.linspace(start=self.margin_size, stop=self.img_w- (self.patch_w + self.margin_size), num=self.patch_num_w).astype(np.float32)-patch_grid_w*0.5
        if self.is_debug:
            start_h_array = np.linspace(start=self.margin_size, stop=self.img_h- (self.patch_h + self.margin_size), num=self.patch_num_h).astype(np.int32).astype(np.float32)
            start_w_array = np.linspace(start=self.margin_size, stop=self.img_w- (self.patch_w + self.margin_size), num=self.patch_num_w).astype(np.int32).astype(np.float32)
        patch_starts = np.array(np.meshgrid(start_w_array, start_h_array)).T.reshape(-1, 2)

        self.patch_grid = np.array([patch_grid_w, patch_grid_h], dtype=np.float32)
        self.patch_infos = np.repeat(patch_starts, repeats = num_img, axis=0)

        

        # repeat the paths according to the number of patches
        self.img_path_list = self.img_path_list * self.num_patch_per_img
        self.key_mask_path_list = self.key_mask_path_list * self.num_patch_per_img
        self.road_mask_path_list = self.road_mask_path_list * self.num_patch_per_img
        # self.overlap_mask_path_list = self.overlap_mask_path_list * self.num_patch_per_img
        self.graph_path_list = self.graph_path_list * self.num_patch_per_img
    
    def random_patch_info(self, patch_info):
        random_translate = np.random.rand(*patch_info.shape).astype(np.float32) * self.patch_grid
        if self.is_debug:
            # random_translate = np.ones_like(random_translate) * self.patch_grid * 0.5
            random_translate = np.zeros_like(random_translate)
        patch_info += random_translate
        start_h = np.clip(patch_info[1], self.margin_size, self.img_h - (self.patch_h + self.margin_size))
        start_w = np.clip(patch_info[0], self.margin_size, self.img_w - (self.patch_w + self.margin_size))
        random_patch_info = np.round(np.array([start_w, start_h])).astype(np.int32)
        return random_patch_info

    def patch_imgs(self, patch_info, *imgs):
        # patch the image and masks
        patch_imgs = []
        for img in imgs:
            patch_img = img[patch_info[1]:patch_info[1]+self.patch_h, patch_info[0]:patch_info[0]+self.patch_w]
            patch_imgs.append(patch_img)
        patch_imgs = tuple(patch_imgs)
        return patch_imgs

    def patch_labels(self, patch_info, label_dict):
        node_points = label_dict['node_points']
        adj_array = label_dict['adj_array']

        if len(node_points) ==0:
            return label_dict
        
        x0, y0 = patch_info[0], patch_info[1]
        x1, y1 = x0 + self.patch_w, y0 + self.patch_h

        node_in_patch_mask = (node_points[:, 0] >= x0) & (node_points[:, 0] < x1) & (node_points[:, 1] >= y0) & (node_points[:, 1] < y1)

        adj_node_mask0 = node_in_patch_mask[adj_array[:, 0]]
        adj_node_mask1 = node_in_patch_mask[adj_array[:, 1]]
        edge_in_patch_mask = adj_node_mask0 | adj_node_mask1

        node_edge_in_patch_mask = np.repeat(edge_in_patch_mask, 2)
        node_indices_in_edges = adj_array.flatten()

        np.maximum.at(node_in_patch_mask, node_indices_in_edges, node_edge_in_patch_mask)

        adj_array, node_points = mask_graph(node_in_patch_mask, adj_array, node_points)

        node_points[:, 0] -= x0
        node_points[:, 1] -= y0

        label_dict['node_points'] = node_points
        label_dict['adj_array'] = adj_array
        return label_dict

    def sample_vectors(self, label_dict):
        node_points = label_dict['node_points']
        adj_array = label_dict['adj_array']
        key_node_mask = label_dict['key_node_mask']

        # sample road_angle_maps
        sample_points, sample_inds, t, sample_valid = sample_road_points_on_patch(node_points, key_node_mask,
                                                                                  adj_array,
                                                                                  density=self.vector_teacher.road_sample_density,
                                                                                  patch_h=self.patch_h,
                                                                                  patch_w=self.patch_w,
                                                                                  max_sample=self.vector_teacher.max_road_samples,
                                                                                  random_sigma=self.vector_teacher.road_random_sigma,
                                                                                  key_thres=8)
        label_dict['road_sample_points'] = sample_points.astype(np.float32)
        label_dict['road_sample_valids'] = sample_valid.astype(np.bool_)
        if len(node_points) == 0 or len(adj_array) == 0:
            label_dict['road_angles'] = np.zeros((self.vector_teacher.max_road_samples, 2), dtype=np.float32)
            label_dict['road_target_points'] = np.zeros((self.vector_teacher.max_road_samples, 2), dtype=np.bool_)
        else:
            angles, angle_valid_mask = get_adj_angle(node_points, adj_array, max_degree=self.max_node_degree)
            road_angles, _ = get_edge_angle(node_points, adj_array, sample_inds, t)
            label_dict['road_angles'] = road_angles.astype(np.float32)
            target_points = get_points_on_edges(node_points, adj_array, sample_inds, t)
            label_dict['road_target_points'] = np.nan_to_num(target_points.astype(np.float32))



        # sample key_angle_maps
        key_points = node_points[key_node_mask]
        sample_points, sample_inds, sample_valid = sample_key_points_on_patch(key_points,
                                                                              density=self.vector_teacher.key_sample_density,
                                                                              patch_h=self.patch_h,
                                                                              patch_w=self.patch_w,
                                                                              max_sample=self.vector_teacher.max_key_samples,
                                                                              random_sigma=self.vector_teacher.key_random_sigma)
        label_dict['key_sample_points'] = sample_points.astype(np.float32)
        label_dict['key_sample_valids'] = sample_valid.astype(np.bool_)
        if len(key_points) == 0:
            label_dict['key_angles'] = np.zeros((self.vector_teacher.max_key_samples, self.max_node_degree),
                                                dtype=np.float32)
            label_dict['key_angle_valids'] = np.zeros((self.vector_teacher.max_key_samples, self.max_node_degree),
                                                      dtype=np.bool_)
            label_dict['key_target_points'] = np.zeros((self.vector_teacher.max_key_samples, 2), dtype=np.float32)
        else:
            key_angles = angles[key_node_mask]
            key_angle_valid_mask = angle_valid_mask[key_node_mask]

            label_dict['key_angles'] = key_angles[sample_inds].astype(np.float32)
            label_dict['key_angle_valids'] = key_angle_valid_mask[sample_inds].astype(np.bool_)
            label_dict['key_target_points'] = key_points[sample_inds].astype(np.float32)
        return label_dict

    def augment(self, img, label_dict):
        node_points = label_dict['node_points']
        masks = label_dict['seg_masks']
        augmented = self.train_augments(image=img, keypoints=node_points.tolist(), mask = masks)
        aug_img = augmented['image']
        aug_node_points = np.array(augmented['keypoints'])
        aug_masks = augmented['mask']
        label_dict['node_points'] = aug_node_points
        label_dict['seg_masks'] = aug_masks
        return aug_img, label_dict

    def __getitem__(self, index):
        if self.is_debug:
            index = index % (self.num_train_data * self.num_patch_per_img)

        img, key_mask, road_mask = self.read_img(index)

        with open(self.graph_path_list[index], 'rb') as f:
            graph_data = pickle.load(f)

        label_dict = graph_data
        patch_info = self.random_patch_info(self.patch_infos[index])
        img, key_mask, road_mask = self.patch_imgs(patch_info, img, key_mask, road_mask)
        label_dict = self.patch_labels(patch_info, label_dict)
        label_dict['seg_masks'] = np.stack((key_mask, road_mask), axis=-1)
        img, label_dict = self.augment(img, label_dict)

        if self.train_vector:
            node_points = label_dict['node_points']
            adj_array = label_dict['adj_array']
            key_node_mask = get_key_node_mask(node_points, adj_array, self.poly_angle_thres)
            label_dict['key_node_mask'] = key_node_mask
            in_range_mask = node_in_patch(node_points, self.patch_h, self.patch_w)
            key_node_mask[~in_range_mask] = False
            label_dict = self.sample_vectors(label_dict)

        label_dict_out = {key: label_dict[key] for key in self.label_keyname_list}
        if self.is_debug:  # debug
            label_dict_out['node_points'] = label_dict['node_points']
            label_dict_out['adj_array'] = label_dict['adj_array']
        return img, label_dict_out
    
    def collate_fn(self, batch):
        imgs = []
        item_dicts = {key: [] for key in batch[0][1].keys()}
        for b in batch:
            img = cv2.cvtColor(b[0], cv2.COLOR_BGR2RGB)
            img = self.transforms(img) if self.transforms else torch.from_numpy(img).permute(2, 0, 1).float()
            imgs.append(img)
            for key in self.label_keyname_list:
                data = b[1][key] 
                if isinstance(data, np.ndarray):
                    item_dicts[key].append(torch.from_numpy(b[1][key]))
        imgs = torch.stack(imgs, dim=0)
        for key in self.label_keyname_list:
            if key == 'seg_masks':
                item_dicts[key] = torch.stack(item_dicts[key], axis=0).permute(0, 3, 1, 2).float()/255.0
            else:
                item_dicts[key] = torch.stack(item_dicts[key], axis=0)
        item_dicts['imgs'] = imgs
        return item_dicts


    
    # def test_getitem(self, index):
    #     os.makedirs('./debug_test/', exist_ok=True)
    #     img_ori, label_dict = self.__getitem__(index)
    #     print(label_dict.keys())
    #
    #     node_points = label_dict['node_points']
    #     adj_array = label_dict['adj_array']
    #
    #
    #     key_node_mask = get_key_node_mask(node_points, adj_array, self.poly_angle_thres)
    #
    #     cv2.imwrite(f'./debug_test/input_img_{str(index)}.png', img_ori)
    #
    #     road_points = node_points[~key_node_mask]
    #     key_points = node_points[key_node_mask]
    #
    #     img = img_ori.copy()
    #     img = np.zeros((self.patch_h, self.patch_w, 3), dtype=np.uint8)
    #     img = plot_edges(node_points, adj_array, img_shape=(self.patch_h, self.patch_w), edge_color=(255, 255, 255), line_width=1, img=img)
    #     img = plot_points(road_points, img_shape=(self.patch_h, self.patch_w), radius=1, point_color=(0, 255, 0), img=img)
    #     img = plot_points(key_points, img_shape=(self.patch_h, self.patch_w), radius=1, point_color=(0, 0, 255), img=img)
    #     cv2.imwrite(f'./debug_test/orig_{str(index)}.png', img)
    #
    #
    #
    #     road_noise_points = label_dict['road_sample_points']
    #     road_target_points = label_dict['road_target_points']
    #     road_noise_valid = label_dict['road_sample_valids']
    #
    #     # norm = np.linalg.norm(road_noise_points - road_target_points, axis=1)[road_noise_valid]
    #     # print(norm)
    #
    #     img = np.zeros((self.patch_h, self.patch_w, 3), dtype=np.uint8)
    #     img = plot_edges(node_points, adj_array, img_shape=(self.patch_h, self.patch_w), edge_color=(255, 255, 255), line_width=1, img=img)
    #     img = plot_points(road_noise_points[road_noise_valid], img_shape=(self.patch_h, self.patch_w), radius=1, point_color=(0, 255, 0), img=img)
    #     img = plot_points(road_target_points[road_noise_valid], img_shape=(self.patch_h, self.patch_w), radius=1, point_color=(255, 0, 0), img=img)
    #     cv2.imwrite(f'./debug_test/road_noise_points_{str(index)}.png', img)
    #
    #
    #     key_noise_points = label_dict['key_sample_points']
    #     key_target_points = label_dict['key_target_points']
    #     key_noise_valid = label_dict['key_sample_valids']
    #     img = np.zeros((self.patch_h, self.patch_w, 3), dtype=np.uint8)
    #     img = plot_edges(node_points, adj_array, img_shape=(self.patch_h, self.patch_w), edge_color=(255, 255, 255), line_width=1, img=img)
    #     img = plot_points(key_noise_points[key_noise_valid], img_shape=(self.patch_h, self.patch_w), radius=1, point_color=(0, 255, 0), img=img)
    #     img = plot_points(key_target_points[key_noise_valid], img_shape=(self.patch_h, self.patch_w), radius=1, point_color=(255, 0, 0), img=img)
    #     cv2.imwrite(f'./debug_test/key_noise_points_{str(index)}.png', img)
    #
    #     road_angle_points = label_dict['road_sample_points']
    #     road_angle_valid = label_dict['road_sample_valids']
    #     road_angles = label_dict['road_angles']
    #
    #     road_angle_points = road_angle_points[road_angle_valid]
    #     road_angles = road_angles[road_angle_valid]
    #     angle_valid = np.ones_like(road_angles, dtype=bool)
    #
    #
    #     img = np.zeros((self.patch_h, self.patch_w, 3), dtype=np.uint8)
    #     img = plot_edges(node_points, adj_array, img_shape=(self.patch_h, self.patch_w), edge_color=(255, 255, 255), img=img)
    #     img = plot_points(road_angle_points, img_shape=(self.patch_h, self.patch_w), radius=2, point_color=(0, 255, 0), img=img)
    #     img = plot_road_angle(road_angle_points, road_angles, angle_valid, img_shape=(self.patch_h, self.patch_w), img=img, line_color=(0, 0, 255), ray_length=10)
    #     cv2.imwrite(f'./debug_test/road_angle_points_{str(index)}.png', img)
    #
    #
    #     key_angle_points = label_dict['key_sample_points']
    #     key_vector_valid = label_dict['key_sample_valids']
    #     sample_angles = label_dict['key_angles']
    #     angle_valid = label_dict['key_angle_valids']
    #
    #     img = np.zeros((self.patch_h, self.patch_w, 3), dtype=np.uint8)
    #     img = plot_edges(node_points, adj_array, img_shape=(self.patch_h, self.patch_w), edge_color=(255, 255, 255), img=img)
    #     img = plot_points(key_angle_points, img_shape=(self.patch_h, self.patch_w), radius=2, point_color=(0, 255, 0), img=img)
    #     img = plot_road_angle(key_angle_points, sample_angles, angle_valid, img_shape=(self.patch_h, self.patch_w), img=img, line_color=(0, 0, 255), ray_length=10)
    #     cv2.imwrite(f'./debug_test/key_angle_points_{str(index)}.png', img)
    #
    #
    #
    #
    #     masks = label_dict['seg_masks']
    #     key_mask = masks[:, :, 0]
    #     road_mask = masks[:, :, 1]
    #     overlap_mask = masks[:, :, 2]
    #
    #     cv2.imwrite(f'./debug_test/key_mask_{str(index)}.png', key_mask)
    #     cv2.imwrite(f'./debug_test/road_mask_{str(index)}.png', road_mask)
    #     cv2.imwrite(f'./debug_test/overlap_mask_{str(index)}.png', overlap_mask)
    #







        

