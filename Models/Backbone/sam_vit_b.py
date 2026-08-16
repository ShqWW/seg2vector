import torch
from torch import nn
from segment_anything.modeling.image_encoder import ImageEncoderViT
from functools import partial
import torch.nn.functional as F

class VitEncoder(nn.Module):
    def __init__(self, cfg):
        super(VitEncoder, self).__init__()
        ### SAM config (B)
        encoder_embed_dim=768
        encoder_depth=12
        encoder_num_heads=12
        encoder_global_attn_indexes=[2, 5, 8, 11]
        prompt_embed_dim = 256
        vit_patch_size = 16

        self.image_encoder = ImageEncoderViT(
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=cfg.patch_h,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=16,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim
            )


        with open(cfg.sam_ckpt, "rb") as f:
            ckpt_state_dict = torch.load(f)

            ## Resize pos embeddings, if needed
            image_size = cfg.patch_h
            if image_size != 1024:
                new_state_dict = self.resize_sam_pos_embed(ckpt_state_dict, image_size, vit_patch_size, encoder_global_attn_indexes)
                ckpt_state_dict = new_state_dict
            
            matched_names = []
            mismatch_names = []
            state_dict_to_load = {}
            for k, v in self.named_parameters():
                if k in ckpt_state_dict and v.shape == ckpt_state_dict[k].shape:
                    matched_names.append(k)
                    state_dict_to_load[k] = ckpt_state_dict[k]
                else:
                    mismatch_names.append(k)
            # print("###### Matched params ######")
            # print(matched_names)
            # print("###### Mismatched params ######")
            # print(mismatch_names)

            self.matched_param_names = set(matched_names)
            self.load_state_dict(state_dict_to_load, strict=False)
        

    def forward(self, img):
        img_features = self.image_encoder(img)
        return img_features
    
    def resize_sam_pos_embed(self, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
        new_state_dict = {k : v for k, v in state_dict.items()}
        pos_embed = new_state_dict['image_encoder.pos_embed']
        token_size = int(image_size // vit_patch_size)
        if pos_embed.shape[1] != token_size:
            # Copied from SAMed
            # resize pos embedding, which may sacrifice the performance, but I have no better idea
            pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
            pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
            new_state_dict['image_encoder.pos_embed'] = pos_embed
            rel_pos_keys = [k for k in state_dict.keys() if 'rel_pos' in k]
            global_rel_pos_keys = [k for k in rel_pos_keys if any([str(i) in k for i in encoder_global_attn_indexes])]
            for k in global_rel_pos_keys:
                rel_pos_params = new_state_dict[k]
                h, w = rel_pos_params.shape
                rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
                rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
                new_state_dict[k] = rel_pos_params[0, 0, ...]
        return new_state_dict

        

