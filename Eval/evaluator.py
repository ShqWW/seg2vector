import numpy as np

import numpy as np
import cv2
import os
import pickle
from utils.graph_utils import transfer_graph_to_dict, get_graph_from_json
from .view_utils import *

class Evaluator():
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.result_path = os.path.join(cfg.result_path, 'graph')
        self.meta_path = os.path.join(cfg.result_path, 'meta_data')
        self.view_path = cfg.view_path
        self.patch_h, self.patch_w = cfg.patch_h, cfg.patch_w
        self.img_h, self.img_w = cfg.img_h, cfg.img_w
        self.dataset = cfg.dataset
        self.is_view = cfg.is_view
        self.dataset_dir = cfg.dataset_dir
        self.in_domain_test = cfg.in_domain_test
        self.scale = 4
        self.img_size = (cfg.img_h, cfg.img_w)
      

        self.get_img_dir()

        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.meta_path, exist_ok=True)
        if self.is_view:
            os.makedirs(self.view_path, exist_ok=True)

    def get_img_dir(self):
        if self.dataset == 'spacenet':
            img_dir = os.path.join(self.dataset_dir, 'RGB_1.0_meter')
        elif self.dataset == 'cityscale':
            img_dir = os.path.join(self.dataset_dir, '20cities')
        elif self.dataset == 'globalscale':
            if self.in_domain_test:
                img_dir = os.path.join(self.dataset_dir, 'in-domain-test')
            else:
                img_dir = os.path.join(self.dataset_dir, 'out_of_domain')
        self.img_dir = img_dir

        
    def save_results(self, sample_dict, result_dict):
        img_name = sample_dict['img_name']
        node_points = result_dict['node_points'].copy()
        adj_array = result_dict['adj_array'].copy()

        if self.dataset == 'spacenet':
            meta_name = img_name.replace('__rgb.png', '')
            node_points[:, 1] = 400 - node_points[:, 1]
        if self.dataset == 'cityscale' or self.dataset == 'globalscale':
            meta_name = img_name.replace('_sat.png', '')
            meta_name = meta_name.replace('region_', '')
        json_results = transfer_graph_to_dict(node_points, adj_array)
        result_path = os.path.join(self.result_path, meta_name + '.p')
        with open(result_path, 'wb') as f:
            pickle.dump(json_results, f)

        meta_dict = result_dict
        meta_dict['meta_name'] = meta_name
        meta_path = os.path.join(self.meta_path, meta_name + '.p')
        with open(meta_path, 'wb') as f:
            pickle.dump(meta_dict, f)

        return result_path, meta_path
    

    def process(self, sample_dict, result_dict):
        result_path, meta_path = self.save_results(sample_dict, result_dict)
        if self.is_view:
            self.view(result_path, meta_path)

    def view(self, result_path, meta_path):
        node_points, adj_array = get_graph_from_json(result_path)
        meta_dict = pickle.load(open(meta_path, 'rb'))
        meta_name = meta_dict['meta_name']

        if self.dataset == 'spacenet':
            img_name = meta_name + '__rgb.png'
            node_points[:, 1] = 400 - node_points[:, 1]
        if self.dataset == 'cityscale' or self.dataset == 'globalscale':
            img_name = 'region_' + meta_name + '_sat.png'
        ori_img = cv2.imread(os.path.join(self.img_dir, img_name))



        # key_points = meta_dict['key_points']
        # road_points = meta_dict['road_points']
        # adj_array = meta_dict['adj_array']
        # node_points = meta_dict['node_points']


        # plot the final results
        graph_img = ori_img.copy()
        node_points = scale_data(node_points, self.scale)
        # key_points = scale_data(key_points, self.scale)
        # road_points = scale_data(road_points, self.scale)
        graph_img = scale_img(graph_img, self.scale)
        graph_img = plot_edges(node_points, adj_array, edge_color=(0, 180, 244), line_width=2*self.scale, img=graph_img)
        graph_img = plot_vertice(node_points, radius=3*self.scale, thickness=1*self.scale, point_color=(0, 180, 244), img=graph_img)
        # graph_img = plot_vertice(road_points, radius=3*self.scale, thickness=1*self.scale, point_color=(0, 180, 244), img=graph_img)
        # graph_img = plot_vertice(key_points, radius=3*self.scale, thickness=1*self.scale, point_color=(80, 80, 255), img=graph_img)
        # graph_img = rescale_img(graph_img, self.scale)
        cv2.imwrite(os.path.join(self.view_path, meta_name+'_graph.png'), graph_img)


        mask_scores = meta_dict['mask_scores']
        key_masks = (mask_scores[0] * 255).astype(np.uint8)
        road_masks = (mask_scores[1] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(self.view_path, meta_name+'_key_mask.png'), key_masks)
        cv2.imwrite(os.path.join(self.view_path, meta_name+'_road_mask.png'), road_masks)


        road_points = meta_dict['road_points']
        key_points = meta_dict['key_points']

        # point_img = ori_img.copy()
        point_img = np.stack((road_masks, road_masks, road_masks), axis=-1).copy()
        point_img = scale_img(point_img, self.scale)
        road_points = scale_data(road_points, self.scale)
        key_points = scale_data(key_points, self.scale)

        point_img = plot_points(key_points, radius=2*self.scale, point_color=(80, 80, 255), img=point_img)
        point_img = plot_points(road_points, radius=2*self.scale, point_color=(0, 180, 244), img=point_img)
        # point_img = rescale_img(point_img, self.scale)
        cv2.imwrite(os.path.join(self.view_path, meta_name+'_point.png'), point_img)


        key_anlges = meta_dict['key_angles']
        road_angles = meta_dict['road_angles']
        key_angle_valids = meta_dict['key_angle_valids']
        road_angle_valids = meta_dict['road_angle_valids']


        # vector_img = ori_img.copy()
        vector_img = np.stack((road_masks, road_masks, road_masks), axis=-1).copy()
        vector_img = scale_img(vector_img, self.scale)



        
        

        vector_img = plot_dual_vectors(road_angles, road_angle_valids, road_points, thickness=1*self.scale, len_vector=6*self.scale, angle_color=(0, 180, 244), img=vector_img)
        vector_img = plot_multi_vectors(key_anlges, key_angle_valids, key_points, thickness=1*self.scale, len_vector=6*self.scale, angle_color=(80, 80, 255), img=vector_img,)
        # vector_img = plot_points(key_points, radius=2*self.scale, point_color=(80, 80, 255), img=vector_img)
        # vector_img = plot_points(road_points, radius=2*self.scale, point_color=(0, 180, 244), img=vector_img)
        # vector_img = rescale_img(vector_img, self.scale)
        cv2.imwrite(os.path.join(self.view_path, meta_name+'_vector.png'), vector_img)
      




    

        

         




