import torch
import math
import torch.nn as nn
import numpy as np
from utils.nms_utils import *
from utils.point_utils import *
from utils.angle_utils import *
# import cv2
# from utils.plot_utils import plot_single_angle_map
# from skimage.measure import label, regionprops

class VectorInferencer(nn.Module):
    def __init__(self, cfg):
        super(VectorInferencer, self).__init__()
        self.img_w, self.img_h = cfg.img_w, cfg.img_h
        self.patch_h, self.patch_w = cfg.patch_h, cfg.patch_w
        self.margin_size = cfg.margin_size
        self.patch_num_h, self.patch_num_w = cfg.patch_num_h, cfg.patch_num_w
        self.patch_num_h_vec, self.patch_num_w_vec = cfg.patch_num_h_vec, cfg.patch_num_w_vec

        start_h_array = np.linspace(start=self.margin_size, stop=self.img_h- (self.patch_h + self.margin_size), num=self.patch_num_h).astype(np.int32)
        start_w_array = np.linspace(start=self.margin_size, stop=self.img_w- (self.patch_w + self.margin_size), num=self.patch_num_w).astype(np.int32)

        start_h_array, start_w_array = torch.from_numpy(start_h_array), torch.from_numpy(start_w_array)
        patch_start_grid = torch.cartesian_prod(start_w_array, start_h_array)
        self.register_buffer(name='patch_start_grid', tensor=patch_start_grid)
        self.patch_num = self.patch_start_grid.shape[0]


        start_h_array = np.linspace(start=self.margin_size, stop=self.img_h- (self.patch_h + self.margin_size), num=self.patch_num_h_vec).astype(np.int32)
        start_w_array = np.linspace(start=self.margin_size, stop=self.img_w- (self.patch_w + self.margin_size), num=self.patch_num_w_vec).astype(np.int32)

        start_h_array, start_w_array = torch.from_numpy(start_h_array), torch.from_numpy(start_w_array)
        patch_start_grid = torch.cartesian_prod(start_w_array, start_h_array)
        self.register_buffer(name='patch_start_grid_vec', tensor=patch_start_grid)
        self.patch_num_vec = self.patch_start_grid_vec.shape[0]


        self.seg_infer = cfg.SEG_INFER
        self.vector_infer = cfg.VECTOR_INFER
        self.num_key_vector_anchors = cfg.num_key_vector_anchors

    def combine_segmap(self, patch_segmaps, is_key=False):
        '''

        :param patch_segmaps: (num_patch, num_seg_maps, patch_h, patch_w)
        :return: combine_segmap: (num_seg_maps, img_h, img_w)
        '''
        num_seg_maps = patch_segmaps.shape[1]

        patch_segmaps = patch_segmaps.view(-1, num_seg_maps, self.patch_h, self.patch_w)
        combine_segmaps = torch.zeros((num_seg_maps, self.img_h, self.img_w), device=patch_segmaps.device)
        count_map = torch.zeros((self.img_h, self.img_w), device=patch_segmaps.device)

        for p in range(self.patch_num_vec if is_key else self.patch_num):
            if is_key:
                start_x, start_y = self.patch_start_grid_vec[p][0].cpu().item(), self.patch_start_grid_vec[p][1].cpu().item()
            else:
                start_x, start_y = self.patch_start_grid[p][0].cpu().item(), self.patch_start_grid[p][1].cpu().item()
            end_x, end_y = start_x + self.patch_w, start_y + self.patch_h
            combine_segmaps[:, start_y:end_y, start_x:end_x] += patch_segmaps[p, :, :, :]
            count_map[start_y:end_y, start_x:end_x] += 1.0

        combine_segmaps = combine_segmaps / count_map.unsqueeze(0)
        return combine_segmaps

    def patch_segmap(self, seg_maps):
        '''
        :param seg_maps: (num_segmaps, H, W)
        :return: patch_seg_maps: (num_patch, num_segmaps, patch_h, patch_w)
        '''
        patch_seg_map_list = []

        for p in range(self.patch_num_vec):
            start_x, start_y = self.patch_start_grid_vec[p][0].cpu().item(), self.patch_start_grid_vec[p][1].cpu().item()
            end_x, end_y = start_x + self.patch_w, start_y + self.patch_h
            patch_seg_map = seg_maps[:, start_y:end_y, start_x:end_x]
            patch_seg_map_list.append(patch_seg_map)

        patch_seg_maps = torch.stack(patch_seg_map_list, dim=0)
        return patch_seg_maps



    def seg2points(self, key_seg, road_seg):
        road_pos_mask = road_seg > self.seg_infer.road_conf_thres
        indices = torch.nonzero(road_pos_mask, as_tuple=False)
        road_points = torch.flip(indices, dims=[-1]).float()
        road_scores = road_seg[road_pos_mask]

        key_pos_mask = (key_seg > self.seg_infer.key_conf_thres) & road_pos_mask
        indices = torch.nonzero(key_pos_mask, as_tuple=False)
        key_points = torch.flip(indices, dims=[-1]).float()
        key_scores = key_seg[key_pos_mask]
        return key_points, road_points, key_scores, road_scores




    def patch_sample_points(self, points):
        points_list = []
        patch_point_inds_list = []

        for p in range(self.patch_num_vec):
            start_x, start_y = self.patch_start_grid_vec[p][0].cpu().item(), self.patch_start_grid_vec[p][1].cpu().item()
            end_x, end_y = start_x + self.patch_w, start_y + self.patch_h

            patch_mask = torch.logical_and(
                torch.logical_and(points[..., 0] >= start_x, points[..., 0] < end_x),
                torch.logical_and(points[..., 1] >= start_y, points[..., 1] < end_y)
            )
            patch_points = points[patch_mask] - self.patch_start_grid_vec[p][None, :]
            patch_points_inds = torch.nonzero(patch_mask, as_tuple=False).squeeze(1)

            points_list.append(patch_points)
            patch_point_inds_list.append(patch_points_inds)

        patch_len = len(points_list)
        max_length = max(patch.shape[0] for patch in points_list)

        for p in range(patch_len):
            points_list[p] = padding_tensor_torch(points_list[p], max_length)
            patch_point_inds_list[p] = padding_tensor_torch(patch_point_inds_list[p],max_length)

        patch_points = torch.stack(points_list, dim=0)
        points_patch_inds = torch.stack(patch_point_inds_list, dim=0)
        return patch_points, points_patch_inds

    def patch_sample_points_single(self, points):
        patch_centers = self.patch_start_grid_vec + torch.tensor([self.patch_w / 2, self.patch_h / 2],
                                                             device=self.patch_start_grid_vec.device)

        dists = torch.cdist(points, patch_centers)
        patch_ids = torch.argmin(dists, dim=1)
        points_list = []
        patch_point_inds_list = []

        for p in range(patch_centers.shape[0]):
            patch_mask = (patch_ids == p)
            patch_points = points[patch_mask] - self.patch_start_grid_vec[p][None, :]
            patch_points_inds = torch.nonzero(patch_mask, as_tuple=False).squeeze(1)
            points_list.append(patch_points)
            patch_point_inds_list.append(patch_points_inds)

        patch_len = len(points_list)
        max_length = max(patch.shape[0] for patch in points_list)

        for p in range(patch_len):
            points_list[p] = padding_tensor_torch(points_list[p], max_length)
            patch_point_inds_list[p] = padding_tensor_torch(patch_point_inds_list[p], max_length)

        patch_points = torch.stack(points_list, dim=0)
        points_patch_inds = torch.stack(patch_point_inds_list, dim=0)
        return patch_points, points_patch_inds

    def combine_tensors(self, tensors, tensor_inds):
        valid_mask = tensor_inds != -1
        tensors = tensors[valid_mask]
        tensor_inds = tensor_inds[valid_mask]
        tensor_dim = tensors.shape[1:]
        if len(tensor_inds) == 0:
            combined_tensor = torch.zeros((0, *tensor_dim), dtype=tensors.dtype, device=tensors.device)
        else:
            combined_tensor_scatter = torch.zeros((tensor_inds.max() + 1, *tensor_dim), dtype=tensors.dtype, device=tensors.device)
            counts = torch.bincount(tensor_inds)
            if len(tensor_dim) > 0:
                counts = counts.view([-1] + [1] * len(tensor_dim)).expand(-1, *tensor_dim)
                tensor_inds = tensor_inds.view([-1] + [1] * len(tensor_dim)).expand(-1, *tensor_dim)
            combined_tensor_scatter.scatter_reduce_(0, tensor_inds, tensors, reduce="sum") 
            combined_tensor = combined_tensor_scatter/counts
        return combined_tensor

    def combine_sample_points(self, patch_points, patch_point_inds):
        patch_start_grid = self.patch_start_grid_vec.unsqueeze(1)
        points = patch_points + patch_start_grid
        points = self.combine_tensors(points, patch_point_inds)

        in_img_mask = torch.logical_and(
            torch.logical_and(points[..., 0] >= self.margin_size,
                              points[..., 0] < self.img_w - self.margin_size),
            torch.logical_and(points[..., 1] >= self.margin_size,
                              points[..., 1] < self.img_h - self.margin_size)
        )

        points = points[in_img_mask]
        return points

    def self_nms(self, points, scores, nms_radius, *args):
        _, keep_mask = point_nms_torch_hd(points, scores, nms_radius)
        points = points[keep_mask]
        scores = scores[keep_mask]
        args = list(args)
        for i in range(len(args)):
            args[i] = args[i][keep_mask]
        return points, scores, *args

    def cross_nms(self, first_points, second_points, second_scores, nms_radius, *args):
        keep_mask = key_nms_torch(second_points, first_points, nms_radius)
        second_points = second_points[keep_mask]
        second_scores = second_scores[keep_mask]
        args = list(args)
        for i in range(len(args)):
            args[i] = args[i][keep_mask]
        return second_points, second_scores, *args

    def combine_key_vectors(self, patch_key_vectors, patch_key_vector_confs, patch_key_inds):
        key_vector_confs = self.combine_tensors(patch_key_vector_confs, patch_key_inds)
        key_vectors = self.combine_tensors(patch_key_vectors, patch_key_inds)
        return key_vectors, key_vector_confs

    def combine_road_vectors(self, patch_road_vectors, patch_road_inds):
        road_vectors = self.combine_tensors(patch_road_vectors, patch_road_inds)
        return road_vectors

    def decode_key_vectors(self, key_vectors, key_vector_confs):
        key_angles = torch.atan2(key_vectors[..., 1], key_vectors[..., 0])
        key_angles[key_angles < 0] += 2 * torch.pi
        key_angles[key_angles >= 2 * torch.pi] -= 2 * torch.pi

        sort_ind = torch.argsort(key_vector_confs, dim=-1, descending=True)
        key_angles = torch.gather(key_angles, dim=-1, index=sort_ind)
        key_angles_conf = torch.gather(key_vector_confs, dim=-1, index=sort_ind)

        key_angle_valids = key_angles_conf > self.vector_infer.key_vector_conf_thres
        if len(key_angle_valids) == 0:
            key_angles = key_angles[:, :1]
            key_angle_valids = torch.zeros_like(key_angles, dtype=torch.bool)
        else:
            max_num_key_angles = torch.sum(key_angle_valids, dim=-1).max()
            key_angles = key_angles[:, :max_num_key_angles]
            key_angle_valids = key_angle_valids[:, :max_num_key_angles]
        return key_angles, key_angle_valids

    def decode_road_vectors(self, road_vectors):
        road_angles = torch.atan2(road_vectors[..., 1], road_vectors[..., 0])
        road_angles[road_angles < 0] += 2 * torch.pi
        road_angles[road_angles >= 2 * torch.pi] -= 2 * torch.pi
        road_angles = road_angles / 2
        road_angles = torch.stack((road_angles, road_angles + math.pi), dim=-1)
        road_anlge_valids = torch.ones_like(road_angles, dtype=torch.bool)
        return road_angles, road_anlge_valids










    


