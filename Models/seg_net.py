import torch
from torch import nn

from .Backbone.build import build_backbone
from .Neck.build import build_neck
from .Head.build import build_seghead

class SegNet(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.neck = build_neck(cfg)
        self.seg_head = build_seghead(cfg)
        self.seg_head_feat_ind = cfg.seg_head_feat_ind
        self.num_patch_infer = cfg.num_patch_infer

    def forward(self, sampled_dict):
        if self.training:
            return self.forward_training(sampled_dict)
        else:
            return self.forward_testing(sampled_dict)

    def feature_extractor(self, sample_dict):
        x = sample_dict['imgs']
        y = self.backbone(x)
        feat = self.neck(y)
        seg_feat = feat[self.seg_head_feat_ind]
        return seg_feat

    def feature_extractor_with_split(self, sample_dict):
        imgs = sample_dict['imgs']
        num_patches = imgs.shape[0]
        sample_dict_splits = []

        for i in range(0, num_patches, self.num_patch_infer):
            end_index = min(i + self.num_patch_infer, num_patches)
            new_sample_dict = {}
            new_sample_dict['imgs'] = imgs[i:end_index]
            sample_dict_splits.append(new_sample_dict)

        seg_feat_list = []
        for sample_dict_split in sample_dict_splits:
            seg_feat = self.feature_extractor(sample_dict_split)
            seg_feat_list.append(seg_feat)
        seg_feat = torch.cat(seg_feat_list, dim=0)
        return seg_feat

    def forward_training(self, sample_dict):
        seg_feats = self.feature_extractor(sample_dict)
        seg_result_dict = self.seg_head(seg_feats)
        return seg_result_dict

    def forward_testing(self, sample_dict):
        seg_feats = self.feature_extractor_with_split(sample_dict)
        seg_result_dict = self.seg_head(seg_feats)
        return seg_result_dict










