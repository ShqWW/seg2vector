import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
import numpy as np


def linear_assign(cost_matrix, gt_valid_mask):
    '''
    cost_matrix: (b, num_pred, num_gt)
    linear_assign 的 Docstring

    :param cost_matrix: 说明
    '''
    cost_matrix_batch = cost_matrix.cpu().numpy()
    gt_valid_mask = gt_valid_mask.cpu().numpy()
    device = cost_matrix.device

    num_batch = cost_matrix_batch.shape[0]
    batch_idx_list = []
    gt_idx_list = []
    pred_idx_list = []

    for b in range(num_batch):
        num_gt = np.sum(gt_valid_mask[b])
        cost_matrix = cost_matrix_batch[b, :, :num_gt]
        mask = np.isnan(cost_matrix)
        cost_matrix[mask] = 1e6
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        batch_idx = torch.full((len(row_ind),), b, dtype=torch.long)
        batch_idx_list.append(batch_idx)
        gt_idx_list.append(torch.tensor(col_ind, dtype=torch.long))
        pred_idx_list.append(torch.tensor(row_ind, dtype=torch.long))

    batch_idx = torch.cat(batch_idx_list, dim=0).to(device)
    gt_idx = torch.cat(gt_idx_list, dim=0).to(device)
    pred_idx = torch.cat(pred_idx_list, dim=0).to(device)

    return batch_idx, pred_idx, gt_idx



class VectorLoss(nn.Module):
    def __init__(self, cfg):
        super(VectorLoss, self).__init__()
        self.road_vector_loss_weight = cfg.road_vector_loss_weight
        self.key_vector_reg_weight = cfg.key_vector_reg_weight
        self.key_vector_cls_weight = cfg.key_vector_cls_weight
        self.key_vector_cost_alpha = cfg.key_vector_cost_alpha

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.smoothl1_loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, pred_dict, label_dict):
        key_sample_valids = label_dict['key_sample_valids']
        road_sample_valids = label_dict['road_sample_valids']

        if road_sample_valids.sum() == 0:
            road_vector_loss = torch.sum(pred_dict['road_vector']) * 0
        else:
            road_vector_preds = pred_dict['road_vector'][road_sample_valids]
            road_angle_targets = label_dict['road_angles'][road_sample_valids]
            road_vector_loss = self.road_vector_loss(road_vector_preds, road_angle_targets)

        if key_sample_valids.sum() == 0:
            key_vector_cls_loss = torch.sum(pred_dict['key_vectors']) * 0
            key_vector_reg_loss = torch.sum(pred_dict['key_vectors']) * 0
        else:
            key_angle_targets = label_dict['key_angles'][key_sample_valids]
            key_angle_valids = label_dict['key_angle_valids'][key_sample_valids]
            key_vector_preds = pred_dict['key_vectors'][key_sample_valids]
            key_vector_logit_preds = pred_dict['key_vectors_logits'][key_sample_valids]
            if key_angle_valids.sum() == 0:
                key_vector_cls_loss = torch.sum(key_vector_logit_preds) * 0
                key_vector_reg_loss = torch.sum(key_vector_preds) * 0
            else:
                key_vector_cls_loss, key_vector_reg_loss = self.key_vector_loss(key_vector_preds,
                                                                                key_vector_logit_preds,
                                                                                key_angle_targets,
                                                                                key_angle_valids)

        total_loss = self.road_vector_loss_weight * road_vector_loss + self.key_vector_cls_weight * key_vector_cls_loss + self.key_vector_reg_weight * key_vector_reg_loss
        loss_dict = {
            'road_vector_loss': road_vector_loss,
            'key_vector_cls_loss': key_vector_cls_loss,
            'key_vector_reg_loss': key_vector_reg_loss
        }
        return total_loss, loss_dict

    def road_vector_loss(self, road_vector_preds, road_angle_targets):
        '''

        :param road_vector_preds: (num_points, 2)
        :param road_angle_targets: (num_points, 2, 2)
        :return:
        '''

        with torch.no_grad():
            road_angle_targets = 2 * road_angle_targets[..., 0] # double the angle to the range of [0, 2pi]
            road_vector_targets = torch.stack([torch.cos(road_angle_targets), torch.sin(road_angle_targets)], dim=-1)

        dot = torch.sum(road_vector_targets * road_vector_preds, dim=-1)
        norm = torch.norm(road_vector_preds, dim=-1)
        loss = 1 - torch.mean(dot / (norm + 1e-6))
        return loss

    def key_vector_loss(self, key_vector_preds, key_vector_logit_preds, key_angle_targets, key_angle_valids):
        with torch.no_grad():
            key_vector_targets = torch.stack([torch.cos(key_angle_targets), torch.sin(key_angle_targets)], dim=-1)
            cos_dist_mat = torch.cosine_similarity(key_vector_preds.unsqueeze(-2), key_vector_targets.unsqueeze(-3),
                                                   dim=-1)
            cos_dist_mat = (cos_dist_mat / 2 + 0.5)
            conf_mat = torch.sigmoid(key_vector_logit_preds).unsqueeze(-1).repeat(1, 1, cos_dist_mat.shape[
                -1])  # (N, num_preds, max_num_targets
            cost_mat = - (self.key_vector_cost_alpha * torch.log(cos_dist_mat + 1e-8) + torch.log(conf_mat + 1e-8))
            batch_id, pos_pred_id, target_id = linear_assign(cost_mat, key_angle_valids)

        conf_targets = torch.zeros_like(key_vector_logit_preds)
        conf_targets[batch_id, pos_pred_id] = 1.0

        cls_loss = self.bce_loss(key_vector_logit_preds, conf_targets)
        matched_key_vector_preds = key_vector_preds[batch_id, pos_pred_id]
        matched_key_vector_targets = key_vector_targets[batch_id, target_id]

        cos_dist_loss = 1 - torch.cosine_similarity(matched_key_vector_preds, matched_key_vector_targets, dim=-1).mean()
        return cls_loss, cos_dist_loss















            


        

