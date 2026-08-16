import torch
from torch import nn

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets):
        smooth = 1e-5
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)
        focal_tversky = (1 - Tversky).pow(self.gamma)

        return focal_tversky

class SegLoss(nn.Module):
    def __init__(self, cfg):
        super(SegLoss, self).__init__()
        self.ftl_loss_weight = cfg.ftl_loss_weight
        self.seg_pos_weight = cfg.seg_pos_weight
        self.ftl_alpha = cfg.ftl_alpha
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.bce_loss_pos = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.seg_pos_weight]), reduction='mean')
        self.ftl_loss = FocalTverskyLoss(alpha=self.ftl_alpha, beta=(1-self.ftl_alpha), gamma=1)

    def forward(self, pred_dict, label_dict):
        seg_mask = label_dict['seg_masks']
        seg_pred = pred_dict['mask_logits']


        seg_mask = seg_mask[:, :2, ...]
        seg_pred = seg_pred[:, :2, ...]

        seg_mask_key = seg_mask[:, 0, ...]
        seg_pred_key = seg_pred[:, 0, ...]
        seg_mask_road = seg_mask[:, 1, ...]
        seg_pred_road = seg_pred[:, 1, ...]

        # For road segmentation loss, only consider pixels with road label as positive samples, and ignore pixels without road label
        seg_pred_key[seg_mask_road == 0] -= 1e6
        seg_mask_key[seg_mask_road == 0] = 0

        road_bce_loss = self.bce_loss_pos(seg_pred_road, seg_mask_road)
        road_ftl_loss = self.ftl_loss(seg_pred_road, seg_mask_road)
        key_seg_loss = self.bce_loss(seg_pred_key, seg_mask_key)


        road_seg_loss = road_bce_loss + self.ftl_loss_weight * road_ftl_loss
        total_loss = key_seg_loss + road_seg_loss

        loss_dict = {
            'road_seg_loss': road_seg_loss,
            'road_ftl_loss': road_ftl_loss,
            'key_seg_loss': key_seg_loss,
        }
        return total_loss, loss_dict


class SegLoss(nn.Module):
    def __init__(self, cfg):
        super(SegLoss, self).__init__()

        self.train_vector = getattr(cfg, 'train_vector', False)

        self.road_bce_loss_weight = cfg.road_bce_loss_weight
        self.road_ftl_loss_weight = cfg.road_ftl_loss_weight

        self.key_bce_loss_weight = cfg.key_bce_loss_weight
        self.key_ftl_loss_weight = cfg.key_ftl_loss_weight

        self.road_bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.road_pos_weight]), reduction='mean')
        self.road_ftl_loss = FocalTverskyLoss(alpha=cfg.road_ftl_alpha, beta=(1-cfg.road_ftl_alpha), gamma=1)

        self.key_bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.key_pos_weight]), reduction='mean')
        self.key_ftl_loss = FocalTverskyLoss(alpha=cfg.key_ftl_alpha, beta=(1-cfg.key_ftl_alpha), gamma=1)

    def forward(self, pred_dict, label_dict):
        seg_mask = label_dict['seg_masks']
        seg_pred = pred_dict['mask_logits']

        seg_mask_key = seg_mask[:, 0, ...]
        seg_pred_key = seg_pred[:, 0, ...]
        seg_mask_road = seg_mask[:, 1, ...]
        seg_pred_road = seg_pred[:, 1, ...]

        # # For road segmentation loss, only consider pixels with road label as positive samples, and ignore pixels without road label
        # seg_pred_key[seg_mask_road == 0] -= 1e6
        # seg_mask_key[seg_mask_road == 0] = 0

        if not self.train_vector:
            road_bce_loss = self.road_bce_loss(seg_pred_road, seg_mask_road)
            road_ftl_loss = self.road_ftl_loss(seg_pred_road, seg_mask_road)
            seg_loss = self.road_bce_loss_weight * road_bce_loss + self.road_ftl_loss_weight * road_ftl_loss
            return seg_loss, {'road_bce_loss': road_bce_loss, 'road_ftl_loss': road_ftl_loss}
        else:
            road_bce_loss = self.road_bce_loss(seg_pred_road, seg_mask_road) # this road bce loss is to monitor if the road seg part is well trained
            key_bce_loss = self.key_bce_loss(seg_pred_key, seg_mask_key)
            key_ftl_loss = self.key_ftl_loss(seg_pred_key, seg_mask_key)
            seg_loss = self.key_bce_loss_weight * key_bce_loss + self.key_ftl_loss_weight * key_ftl_loss
            return seg_loss, {'road_bce_loss': road_bce_loss, 'key_bce_loss': key_bce_loss, 'key_ftl_loss': key_ftl_loss}
