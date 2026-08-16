import numpy as np
import torch
from torch import _is_functional_tensor, cos_, nn
import torch.nn.functional as F
import math

class VectorDecoder(nn.Module):
    def __init__(self, cfg):
        super(VectorDecoder, self).__init__()
        self.patch_h, self.patch_w = cfg.patch_h, cfg.patch_w
        self.in_channels = cfg.seg_enc_channels
        self.inter_channels = cfg.vector_dec_inter_channels
        self.num_patch_infer = cfg.num_patch_infer
        self.num_key_vector_anchors = cfg.num_key_vector_anchors

        key_vector_base_angles = torch.linspace(0, 2*math.pi, steps=self.num_key_vector_anchors+1)[:-1]
        self.register_buffer('key_vector_base_angles', key_vector_base_angles)

        self.feature_extractor_key = nn.Sequential(
            nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=3, padding=1),)


        self.feature_extractor_road = nn.Sequential(
            nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=3, padding=1),)

        self.road_vector_layer = nn.Sequential(nn.Linear(self.inter_channels, self.inter_channels),
                                        nn.ReLU(),
                                        nn.Linear(self.inter_channels, self.inter_channels),
                                        nn.ReLU(),
                                        nn.Linear(self.inter_channels, 2))

        self.key_angle_layer = nn.Sequential(nn.Linear(self.inter_channels, self.inter_channels),
                                        nn.ReLU(),
                                        nn.Linear(self.inter_channels, self.inter_channels),
                                        nn.ReLU(),
                                        nn.Linear(self.inter_channels, self.num_key_vector_anchors * 2)
                                        )

    def init_wieght(self):
        for p in self.key_angle_layer2.parameters():
            nn.init.normal_(p, mean=0., std=1e-3)

    def point_feat_sampler(self, featmap, points):
        '''
        Input:
        featmap: (b, c, h, w)
        points: (b, n, 2)
        Return:
        feature: (b, n, c)

        '''
        feat = featmap
        points_x = (points[..., 0] / self.patch_h) *2 - 1
        points_y = (points[..., 1] / self.patch_w) *2 - 1
        points = torch.stack([points_x, points_y], dim=-1).unsqueeze(2)
        features = F.grid_sample(feat, points, mode='bilinear', align_corners=False).squeeze(-1).permute(0, 2, 1)
        return features

    def get_key_pool_offsets(self):
        cos_angles = torch.cos(self.key_vector_base_angles)
        sin_angles = torch.sin(self.key_vector_base_angles)
        point_offsets = torch.linspace(0, self.key_pool_edge_length, steps=self.num_pool_points)
        offset_x = point_offsets.unsqueeze(0) * cos_angles.unsqueeze(1)
        offset_y = point_offsets.unsqueeze(0) * sin_angles.unsqueeze(1)
        key_pool_offset = torch.stack([offset_x, offset_y], dim=-1) # (num_anchors, num_pool_points, 2)
        return key_pool_offset


    def road_vectors_detector(self, feats, road_vector_points):
        feats = self.feature_extractor_road(feats)
        road_points_feats = self.point_feat_sampler(feats, road_vector_points)
        road_vectors = self.road_vector_layer(road_points_feats)
        road_vectors = road_vectors.view(*road_vectors.shape[:-1], 2)
        return road_vectors

    def key_vectors_detector(self, feats, key_vector_points):
        feats = self.feature_extractor_key(feats)
        key_points_feats = self.point_feat_sampler(feats, key_vector_points)
        key_angles_out = self.key_angle_layer(key_points_feats)  # (B, N_key, 2*num_anchors)
        # print(aaa)

        key_angles_out = key_angles_out.view(*key_angles_out.shape[:-1], self.num_key_vector_anchors, 2)
        key_vector_logits = key_angles_out[..., 0]  # (B, N_key, num_anchors)
        key_angle_offsets = key_angles_out[..., 1]
        key_angles = self.key_vector_base_angles.unsqueeze(0).unsqueeze(0) + key_angle_offsets
        key_vectors = torch.stack([torch.cos(key_angles), torch.sin(key_angles)], dim=-1)  # (B, N_key, num_anchors, 2)
        return key_vector_logits, key_vectors

    def forward(self, feats, sample_dicts):
        key_sample_points = sample_dicts['key_sample_points']  # (B, N_key_angle, 2)
        road_sample_points = sample_dicts['road_sample_points']  # (B, N_road_angle, 2)

        key_vector_logits, key_vectors = self.key_vectors_detector(feats, key_sample_points)
        road_vectors = self.road_vectors_detector(feats, road_sample_points)

        result_dict = {
            'key_vectors': key_vectors,
            'key_vectors_logits': key_vector_logits,
            'road_vector': road_vectors,
        }

        return result_dict

    def infer_road(self, feats, road_points):
        bs = feats.shape[0]
        road_vectors_list = []
        num_splits = bs // self.num_patch_infer + 1


        for i in range(num_splits):
            start_idx = i * self.num_patch_infer
            end_idx = min((i + 1) * self.num_patch_infer, bs)

            feats_split = feats[start_idx:end_idx, ...]
            road_angle_points_split = road_points[start_idx:end_idx, ...]

            road_vectors = self.road_vectors_detector(feats_split, road_angle_points_split)
            road_vectors_list.append(road_vectors)

        road_vectors = torch.cat(road_vectors_list, dim=0)
        return road_vectors


    def infer_key(self, feats, key_points):
        bs = feats.shape[0]
        key_vectors_list = []
        key_vector_conf_list = []
        num_splits = bs // self.num_patch_infer + 1


        for i in range(num_splits):
            start_idx = i * self.num_patch_infer
            end_idx = min((i + 1) * self.num_patch_infer, bs)

            feats_split = feats[start_idx:end_idx, ...]
            key_angle_points_split = key_points[start_idx:end_idx, ...]

            key_vector_logits, key_vectors = self.key_vectors_detector(feats_split, key_angle_points_split)
            key_vector_conf = torch.sigmoid(key_vector_logits)

            key_vectors_list.append(key_vectors)
            key_vector_conf_list.append(key_vector_conf)

        key_vectors = torch.cat(key_vectors_list, dim=0)
        key_vector_confs = torch.cat(key_vector_conf_list, dim=0)
        return key_vectors, key_vector_confs


    
     
        


        
        
