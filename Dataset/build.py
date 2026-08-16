from torchvision import transforms
# from .img_transfroms import *
transform_all = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # RGB 通道的均值
    std=[0.229, 0.224, 0.225]    # RGB 通道的标准差
)
])

def build_trainset(cfg):
    if cfg.dataset == 'cityscale':
        from .cityscale_dataset import CityscaleTrSet
        trainset = CityscaleTrSet(cfg=cfg, transforms=transform_all)
    elif cfg.dataset == 'spacenet':
        from .spacenet_dataset import SpacenetTrSet
        trainset = SpacenetTrSet(cfg=cfg, transforms=transform_all)
    elif cfg.dataset == 'globalscale':
        from .globalscale_dataset import GlobalscaleTrSet
        trainset = GlobalscaleTrSet(cfg=cfg, transforms=transform_all)
    return trainset




def build_testset(cfg):
    if cfg.dataset == 'cityscale':
        from .cityscale_dataset import CityscaleTsSet
        trainset = CityscaleTsSet(cfg=cfg, transforms=transform_all)
    elif cfg.dataset == 'spacenet':
        from .spacenet_dataset import SpacenetTsSet
        trainset = SpacenetTsSet(cfg=cfg, transforms=transform_all)
    elif cfg.dataset == 'globalscale':
        from .globalscale_dataset import GlobalscaleTsSet
        trainset = GlobalscaleTsSet(cfg=cfg, transforms=transform_all)
    return trainset