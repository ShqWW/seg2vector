def build_seghead(cfg):
    if cfg.seg_head == 'seg_head':
        from .seg_head import SegHead
        seg_head = SegHead(cfg = cfg)
        return seg_head
    elif cfg.seg_head == 'seg_headv2':
        from .seg_headv2 import SegHead
        seg_head = SegHead(cfg = cfg)
        return seg_head
    else:
        return None