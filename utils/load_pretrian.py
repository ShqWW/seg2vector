import torch
# def load_pretrain(net):
#     print('loading the pretriained model from spacenet.ckpt')
#     weight_backbone = {}
#     weight_seg = {}
#
#     weight = torch.load('./spacenet.ckpt', map_location='cpu')['state_dict']
#     for key in weight.keys():
#         if 'image_encoder' in key:
#             weight_backbone[key] = weight[key]
#         if 'map_decoder' in key:
#             new_key = key.replace('map_decoder', 'map_decoder2')
#             weight_seg[new_key] = weight[key]
#     net.seg_net.backbone.load_state_dict(weight_backbone, strict=True)
#     net.seg_net.seg_head.load_state_dict(weight_seg, strict=False)
#     return net


def load_segnet_weight(net, weight_path, ids):
    print('loading the pretrained model with weighted averaging')

    checkpoints = []
    for idx in ids:
        file_path = f"{weight_path}/para_{idx}.pth"
        checkpoints.append((file_path, 1.0))

    loaded_weights = []

    # 1. 加载所有权重文件
    for path, weight_factor in checkpoints:
        print(f"Loading weights from: {path}")
        try:
            ckpt = torch.load(path, map_location='cpu')
            loaded_weights.append(ckpt)
        except FileNotFoundError:
            print(f"Error: File not found at {path}. Please check your net_path.")
            return net

    if not loaded_weights:
        print("Error: No weights were loaded.")
        return net
    averaged_weight = {}
    ref_keys = [k for k in loaded_weights[0].keys() if 'seg_net' in k]
    print(f"Found {len(ref_keys)} keys to average.")
    for key in ref_keys:
        sum_w = None
        for i, w_dict in enumerate(loaded_weights):
            # 确保键存在于所有模型中
            if key in w_dict:
                val = w_dict[key]
                if sum_w is None:
                    sum_w = val * checkpoints[i][1]  # 乘以该文件的权重系数
                else:
                    sum_w += val * checkpoints[i][1]
            else:
                print(f"Warning: Key {key} not found in checkpoint {i}")

        if sum_w is not None:
            total_weight_factor = sum(c[1] for c in checkpoints)
            averaged_weight[key] = sum_w / total_weight_factor

    net.load_state_dict(averaged_weight, strict=False)

    print('Weighted pretrained model loaded successfully.')
    return net

def load_vectorhead_weight(net, weight_path, ids):
    print('loading the pretrained model with weighted averaging')

    checkpoints = []
    for idx in ids:
        file_path = f"{weight_path}/para_{idx}.pth"
        checkpoints.append((file_path, 1.0))

    loaded_weights = []

    # 1. 加载所有权重文件
    for path, weight_factor in checkpoints:
        print(f"Loading weights from: {path}")
        try:
            ckpt = torch.load(path, map_location='cpu')
            loaded_weights.append(ckpt)
        except FileNotFoundError:
            print(f"Error: File not found at {path}. Please check your net_path.")
            return net

    if not loaded_weights:
        print("Error: No weights were loaded.")
        return net
    averaged_weight = {}
    ref_keys = [
        k for k in loaded_weights[0].keys()
        if ('seg_encoder' in k or 'seg_decoder' in k or 'point_decoder' in k or 'vector_decoder' in k)
    ]
    print(f"Found {len(ref_keys)} keys to average.")
    for key in ref_keys:
        sum_w = None
        for i, w_dict in enumerate(loaded_weights):
            # 确保键存在于所有模型中
            if key in w_dict:
                val = w_dict[key]
                if sum_w is None:
                    sum_w = val * checkpoints[i][1]  # 乘以该文件的权重系数
                else:
                    sum_w += val * checkpoints[i][1]
            else:
                print(f"Warning: Key {key} not found in checkpoint {i}")

        if sum_w is not None:
            total_weight_factor = sum(c[1] for c in checkpoints)
            averaged_weight[key] = sum_w / total_weight_factor

    net.load_state_dict(averaged_weight, strict=False)

    print('Weighted pretrained model loaded successfully.')
    return net