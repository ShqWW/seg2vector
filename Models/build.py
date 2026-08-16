from .seg2vector import Seg2Vec
def build_model(cfg):
    model = Seg2Vec(cfg=cfg)
    return model

