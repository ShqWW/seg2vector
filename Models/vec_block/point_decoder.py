import torch
from torch import nn
import torch.nn.functional as F


class PointDecoder(nn.Module):
    def __init__(self, cfg):
        super(PointDecoder, self).__init__()
        self.patch_h, self.patch_w = cfg.patch_h, cfg.patch_w
        self.in_channels = cfg.seg_enc_channels
        self.inter_channels = cfg.point_dec_inter_channels
        self.num_patch_infer = cfg.num_patch_infer

        self.feture_extractor_road = nn.Sequential(
            nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=3, padding=1),)


        self.feture_extractor_key = nn.Sequential(
            nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=3, padding=1),)

        self.road_layer = nn.Sequential(nn.Linear(self.inter_channels, self.inter_channels),
                                        nn.ReLU(),
                                        nn.Linear(self.inter_channels, self.inter_channels),
                                        nn.ReLU(),
                                        nn.Linear(self.inter_channels, 2))


        self.key_layer = nn.Sequential(nn.Linear(self.inter_channels, self.inter_channels),
                                        nn.ReLU(),
                                        nn.Linear(self.inter_channels, self.inter_channels),
                                        nn.ReLU(),
                                        nn.Linear(self.inter_channels, 2))

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


    def road_point_detector(self, feats, road_sample_points):
        feats = self.feture_extractor_road(feats)  # (B, C, H, W)
        road_point_feats = self.point_feat_sampler(feats, road_sample_points)  # (B, N_road_angle, C)
        road_points_offset = self.road_layer(road_point_feats)  # (B, N
        road_sample_points = road_sample_points + road_points_offset  # (B, N_road_angle, num_angle_bins)
        return road_sample_points

    def key_point_detector(self, feats, road_sample_points):
        feats = self.feture_extractor_key(feats)  # (B, C, H, W)
        key_point_feats = self.point_feat_sampler(feats, road_sample_points)  # (B, N_road_angle, C)
        key_points_offset = self.key_layer(key_point_feats)  # (B, N
        key_sample_points = road_sample_points + key_points_offset  # (B, N_road_angle, num_angle_bins)
        return key_sample_points
    
    def forward(self, feats, sample_dicts):
        key_sample_points = sample_dicts['key_sample_points']  # (B, N_key_angle, 2)
        road_sample_points = sample_dicts['road_sample_points']  # (B, N_road_angle, 2)

        key_sample_points = self.key_point_detector(feats, key_sample_points)
        road_sample_points = self.road_point_detector(feats, road_sample_points)

        result_dict = {
            'key_sample_points': key_sample_points,
            'road_sample_points': road_sample_points,
        }
        return result_dict

    def infer_road(self, feats, road_points):
        '''
        :param feats: (B, C, H, W)
        :param road_points: (B, num_road_points, 2)
        :return: road_points: (B, num_road_points, 2)
        '''
        bs = feats.shape[0]
        road_points_list = []
        num_splits = bs // self.num_patch_infer + 1

        for i in range(num_splits):
            start_idx = i * self.num_patch_infer
            end_idx = min((i + 1) * self.num_patch_infer, bs)

            feats_split = feats[start_idx:end_idx, ...]
            road_points_split = road_points[start_idx:end_idx, ...]

            road_points_refine = self.road_point_detector(feats_split, road_points_split)
            road_points_list.append(road_points_refine)

        road_points = torch.cat(road_points_list, dim=0)
        return road_points

    def infer_key(self, feats, key_points):
        '''
        :param feats: (B, C, H, W)
        :param key_points: (B, num_key_points, 2)
        :return: key_points: (B, num_key_points, 2)
        '''
        bs = feats.shape[0]
        key_points_list = []
        num_splits = bs // self.num_patch_infer + 1

        for i in range(num_splits):
            start_idx = i * self.num_patch_infer
            end_idx = min((i + 1) * self.num_patch_infer, bs)

            feats_split = feats[start_idx:end_idx, ...]
            key_points_split = key_points[start_idx:end_idx, ...]

            key_points_refine = self.key_point_detector(feats_split, key_points_split)
            key_points_refine[..., 0] = torch.clamp(key_points_refine[..., 0], min=0, max=self.patch_h-1)
            key_points_refine[..., 1] = torch.clamp(key_points_refine[..., 1], min=0, max=self.patch_w-1)
            key_points_list.append(key_points_refine)
        key_points = torch.cat(key_points_list, dim=0)
        return key_points