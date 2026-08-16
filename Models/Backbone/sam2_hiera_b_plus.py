import torch
from torch import nn
from sam2.build_sam import build_sam2
from sam2.modeling.backbones.hieradet import Hiera
from functools import partial
import torch.nn.functional as F


class VitEncoder(nn.Module):
    def __init__(self, cfg):
        super(VitEncoder, self).__init__()
        self.image_encoder = Hiera(embed_dim=112, num_heads=2)

        state_dict = torch.load(cfg.sam_ckpt, map_location='cpu')['model']

        new_state_dict = {}
        for key in state_dict.keys():
            if 'image_encoder.trunk' in key:
                new_key = key.replace('image_encoder.trunk', 'image_encoder')
                new_state_dict[new_key] = state_dict[key]
        self.load_state_dict(new_state_dict, strict=True)

    def forward(self, img):
        img_features = self.image_encoder(img)
        return img_features


