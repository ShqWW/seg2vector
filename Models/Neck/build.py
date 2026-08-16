def build_neck(cfg):
    if cfg.neck == 'fpn':
        from .fpn import FPN
        head = FPN(cfg)


    if cfg.neck == 'rfpn':
        from .rfpn import RFPN
        head = RFPN(cfg)
    elif cfg.neck == 'blank_neck':
        from .blankneck import BlankNeck
        head = BlankNeck(cfg)
    return head 