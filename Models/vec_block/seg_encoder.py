import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.conv(x)
        out = x + residual
        return self.relu(out)


class SegEncoder(nn.Module):
    def __init__(self, cfg):
        super(SegEncoder, self).__init__()
        in_channels = 1
        self.out_channels = cfg.seg_enc_channels
        self.num_patch_infer = cfg.num_patch_infer

        self.stage1 = self._make_stage(in_channels, 64)
        self.stage2 = self._make_stage(64, 128)

        self.out_layer = nn.Sequential(
            nn.Conv2d(128, self.out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    def _make_stage(self, in_c, out_c):
        stage = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=2, stride=2, padding=0),
                                    nn.ReLU(),
                                    ResidualBlock(out_c))
        return stage

    def encode(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.out_layer(x)
        return x
    
    def encode_with_split(self, x):
        bs = x.shape[0]
        num_splits = bs // self.num_patch_infer + 1
        feat_list = []
        for i in range(num_splits):
            start_idx = i * self.num_patch_infer
            end_idx = min((i + 1) * self.num_patch_infer, bs)
            x_split = x[start_idx:end_idx, ...]
            feat_split = self.encode(x_split)
            feat_list.append(feat_split)
        feats = torch.cat(feat_list, dim=0)
        return feats

    def forward(self, x):
        if self.training:
            return self.encode(x)
        else:
            return self.encode_with_split(x)


        
        



