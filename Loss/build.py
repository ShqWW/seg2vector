def build_loss(cfg):
    from .overall_loss import OverallLoss
    lossfun = OverallLoss(cfg)
    return lossfun