from torch import nn
from .seg_loss import SegLoss
from .point_loss import PointLoss
from .vector_loss import VectorLoss

class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.seg_loss = SegLoss(cfg)
        self.point_loss = PointLoss(cfg)
        self.vector_loss = VectorLoss(cfg)
        self.train_vector = cfg.train_vector

    def forward(self, pred_dict, label_dict):
        loss_dict = {}
        seg_loss, seg_loss_dict = self.seg_loss(pred_dict, label_dict)
        loss_dict.update(seg_loss_dict)
        if self.train_vector:
            point_loss, point_loss_dict = self.point_loss(pred_dict, label_dict)
            vector_loss, vector_loss_dict = self.vector_loss(pred_dict, label_dict)
            loss = seg_loss + point_loss + vector_loss
            loss_dict.update(point_loss_dict)
            loss_dict.update(vector_loss_dict)
        else:
            loss = seg_loss
        loss_dict['total_loss'] = loss
        return loss, loss_dict

    