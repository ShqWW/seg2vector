def build_backbone(cfg):
    if cfg.backbone == 'sam_vit_b':
        from .sam_vit_b import VitEncoder
        backbone = VitEncoder(cfg = cfg)
        return backbone
    elif cfg.backbone == 'sam2_hiera_b+':
        from .sam2_hiera_b_plus import VitEncoder
        backbone = VitEncoder(cfg = cfg)
        return backbone
    else:
        return None