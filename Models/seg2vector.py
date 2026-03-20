import torch
from torch import nn
from .seg_net import SegNet
from .vec_block.seg_encoder import SegEncoder
from .vec_block.seg_decoder import SegDecoder
from .vec_block.point_decoder import PointDecoder
from .vec_block.vector_decoder import VectorDecoder
from .vector_inferencer import VectorInferencer
from .vector2graph import Vector2Graph

class Seg2Vec(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.seg_net = SegNet(cfg)
        self.seg_encoder = SegEncoder(cfg)
        self.seg_decoder = SegDecoder(cfg)
        self.point_decoder = PointDecoder(cfg)
        self.vector_decoder = VectorDecoder(cfg)
        self.vector_inferencer = VectorInferencer(cfg)
        self.topo_inferencer = Vector2Graph(cfg)
        self.seg_infer = cfg.SEG_INFER
        self.train_vector = getattr(cfg, 'train_vector', False)

        if self.train_vector:
            self.freeze_seg()
        else:
            self.freeze_head()

    def forward(self, sample_dict):
        if self.training:
            return self.forward_training(sample_dict)
        else:
            return self.forward_testing(sample_dict)


    def freeze_bn_and_stats(self, module):
        module.eval()
        # 如果该模块有 children (例如 backbone 下面有很多层)，递归处理
        for m in module.children():
            self.freeze_bn_and_stats(m)

    def freeze_seg(self):
        print("Freezing segnet")
        modules_to_freeze = [self.seg_net]

        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
            self.freeze_bn_and_stats(module)


    def freeze_head(self):
        print("Freezing head")
        modules_to_freeze = [self.seg_encoder, self.seg_decoder, self.point_decoder, self.vector_decoder]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
            self.freeze_bn_and_stats(module)

    def forward_training(self, sample_dict):
        road_mask_logits = self.seg_net(sample_dict)
        if self.train_vector:
            road_mask_scores = torch.sigmoid(road_mask_logits).detach()
            feats = self.seg_encoder(road_mask_scores)
            point_dict = self.point_decoder(feats, sample_dict)
            vector_dict = self.vector_decoder(feats, sample_dict)

            key_mask_logits = self.seg_decoder(feats)

        else:
            key_mask_logits = torch.zeros_like(road_mask_logits)

        mask_logits = torch.cat([key_mask_logits, road_mask_logits], dim=1)
        result_dict = {
            'mask_logits': mask_logits,
        }

        if self.train_vector:
            result_dict.update(point_dict)
            result_dict.update(vector_dict)

        return result_dict

    def forward_testing(self, sample_dict):
        road_scores_seg_patch = self.seg_net(sample_dict)
        road_scores_seg = self.vector_inferencer.combine_segmap(road_scores_seg_patch)

        road_scores_patch = self.vector_inferencer.patch_segmap(road_scores_seg)
        feat = self.seg_encoder(road_scores_patch)


        key_scores_seg_patch = self.seg_decoder(feat)
        key_scores_seg = self.vector_inferencer.combine_segmap(key_scores_seg_patch)

        key_scores_seg = key_scores_seg.squeeze(0).squeeze(0)
        road_scores_seg = road_scores_seg.squeeze(0).squeeze(0)

        key_points, road_points, key_scores, road_scores = self.vector_inferencer.seg2points(key_scores_seg, road_scores_seg)

        key_points, key_scores = self.vector_inferencer.self_nms(key_points, key_scores, self.seg_infer.kk_nms_radius)
        road_points, road_scores = self.vector_inferencer.cross_nms(key_points, road_points, road_scores, self.seg_infer.kr_nms_radius)
        road_points, road_scores = self.vector_inferencer.self_nms(road_points, road_scores, self.seg_infer.rr_nms_radius)

        for _ in range(1):
            patch_road_points, patch_road_point_inds = self.vector_inferencer.patch_sample_points(road_points)
            patch_road_points = self.point_decoder.infer_road(feat, patch_road_points)
            road_points = self.vector_inferencer.combine_sample_points(patch_road_points, patch_road_point_inds)

        for _ in range(1):
            patch_key_points, patch_key_point_inds = self.vector_inferencer.patch_sample_points(key_points)
            patch_key_points = self.point_decoder.infer_key(feat, patch_key_points)
            key_points = self.vector_inferencer.combine_sample_points(patch_key_points, patch_key_point_inds)


        patch_road_points, patch_road_point_inds = self.vector_inferencer.patch_sample_points(road_points)
        patch_key_points, patch_key_point_inds = self.vector_inferencer.patch_sample_points(key_points)

        patch_key_vectors, patch_key_vector_confs = self.vector_decoder.infer_key(feat, patch_key_points)
        patch_road_vectors = self.vector_decoder.infer_road(feat, patch_road_points)

        key_vectors, key_vector_confs = self.vector_inferencer.combine_key_vectors(patch_key_vectors, patch_key_vector_confs, patch_key_point_inds)
        road_vectors = self.vector_inferencer.combine_road_vectors(patch_road_vectors, patch_road_point_inds)

        key_angles, key_angle_valids = self.vector_inferencer.decode_key_vectors(key_vectors, key_vector_confs)
        road_angles, road_angle_valids = self.vector_inferencer.decode_road_vectors(road_vectors)

        node_points, adj_array = self.topo_inferencer(key_points, road_points, key_angles, road_angles, key_angle_valids, road_angle_valids)

        seg_masks = torch.stack((key_scores_seg, road_scores_seg), dim=0)
        result_dict = {
            'node_points': node_points,
            'adj_array': adj_array,
            'key_points': key_points,
            'road_points': road_points,
            'key_angles': key_angles,
            'road_angles': road_angles,
            'key_angle_valids': key_angle_valids,
            'road_angle_valids': road_angle_valids,
            'mask_scores': seg_masks,
        }
        return result_dict



