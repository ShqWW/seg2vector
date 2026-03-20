import torch
from torch import nn
from segment_anything.modeling.common import LayerNorm2d
'''
Segmentation head for road segmentation
'''

class SegHead(nn.Module):
    def __init__(self, cfg):
        super(SegHead, self).__init__()
        self.in_channels = cfg.seg_head_in_channels
        self.num_patch_infer = cfg.num_patch_infer

        # the same as the samroad head
        self.map_decoder = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, 128, kernel_size=2, stride=2),
            LayerNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
        )

    def forward_training(self, feats):
        mask_logits = self.map_decoder(feats)
        return mask_logits
    
    def forward_testing(self, feats):
        bs = feats.shape[0]
        num_splits = bs//self.num_patch_infer + 1
        mask_scores_list = []

        for i in range(num_splits):
            start_idx = i * self.num_patch_infer
            end_idx = min((i + 1) * self.num_patch_infer, bs)
            feats_split = feats[start_idx:end_idx, ...]
            
            mask_logits_split = self.map_decoder(feats_split)
            mask_scores_split = torch.sigmoid(mask_logits_split)
            mask_scores_list.append(mask_scores_split)
        mask_scores = torch.cat(mask_scores_list, dim=0)
        return mask_scores
    
    def forward(self, feats):
        if self.training:
            return self.forward_training(feats)
        else:
            return self.forward_testing(feats)
