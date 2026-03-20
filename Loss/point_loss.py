import torch
from torch import nn

class PointLoss(nn.Module):
    def __init__(self, cfg):
        super(PointLoss, self).__init__()
        self.reg_loss_weight = cfg.point_reg_loss_weight
        self.reg_loss = nn.SmoothL1Loss(reduction='mean')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.bce_base_loss = nn.BCELoss(reduction='mean')
        
    def forward(self, pred_dict, label_dict):
        key_sample_valids = label_dict['key_sample_valids']
        road_sample_valids = label_dict['road_sample_valids']

        key_reg_targets = label_dict['key_target_points'][key_sample_valids]
        road_reg_targets = label_dict['road_target_points'][road_sample_valids]

        key_reg_preds = pred_dict['key_sample_points'][key_sample_valids]
        road_reg_preds = pred_dict['road_sample_points'][road_sample_valids]

        key_reg_loss = self.reg_loss(key_reg_preds, key_reg_targets)
        road_reg_loss = self.reg_loss(road_reg_preds, road_reg_targets)

        point_reg_loss = key_reg_loss + road_reg_loss

        total_loss = self.reg_loss_weight * point_reg_loss
        loss_dict = {
            'point_reg_loss': point_reg_loss,
        }

        return total_loss, loss_dict

    def forward(self, pred_dict, label_dict):
        key_sample_valids = label_dict['key_sample_valids']
        road_sample_valids = label_dict['road_sample_valids']

        if key_sample_valids.sum() == 0:
            key_reg_loss = torch.sum(pred_dict['key_sample_points']) * 0
        else:
            key_reg_targets = label_dict['key_target_points'][key_sample_valids]
            key_reg_preds = pred_dict['key_sample_points'][key_sample_valids]
            key_reg_loss = self.reg_loss(key_reg_preds, key_reg_targets)

        if road_sample_valids.sum() == 0:
            road_reg_loss = torch.sum(pred_dict['road_sample_points']) * 0
        else:
            road_reg_targets = label_dict['road_target_points'][road_sample_valids]
            road_reg_preds = pred_dict['road_sample_points'][road_sample_valids]
            road_reg_loss = self.reg_loss(road_reg_preds, road_reg_targets)

        point_reg_loss = key_reg_loss + road_reg_loss
        total_loss = self.reg_loss_weight * point_reg_loss
        loss_dict = {
            'point_reg_loss': point_reg_loss,
            'road_reg_loss': road_reg_loss,
            'key_reg_loss': key_reg_loss,
        }
        return total_loss, loss_dict