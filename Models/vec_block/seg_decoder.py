import torch
import torch.nn as nn
'''
Decode key segmentation masks
'''

class SegDecoder(nn.Module):
    def __init__(self, cfg):
        super(SegDecoder, self).__init__()
        self.channels = cfg.seg_enc_channels
        inter_channels = cfg.seg_dec_inter_channels
        self.num_patch_infer = cfg.num_patch_infer


        self.map_decoder = nn.Sequential(
            nn.Conv2d(self.channels, inter_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1),
            nn.ConvTranspose2d(inter_channels, 1, kernel_size=4, stride=4, padding=0)
        )

    def decode(self, x):
        x = self.map_decoder(x)
        return x
    
    def decode_with_split(self, x):
        bs = x.shape[0]
        num_splits = bs // self.num_patch_infer + 1
        key_scores_list = []
        for i in range(num_splits):
            start_idx = i * self.num_patch_infer
            end_idx = min((i + 1) * self.num_patch_infer, bs)
            x_split = x[start_idx:end_idx, ...]
            key_logits = self.decode(x_split)
            key_scores = torch.sigmoid(key_logits)
            key_scores_list.append(key_scores)
        key_scores = torch.cat(key_scores_list, dim=0)
        return key_scores

    def forward(self, x):
        if self.training:
            return self.decode(x)
        else:
            return self.decode_with_split(x)


        
        



