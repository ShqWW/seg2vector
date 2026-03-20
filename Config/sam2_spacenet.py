from types import SimpleNamespace
import math

# dataset
random_seed = 4016
dataset = 'spacenet'
dataset_dir = '/yourdataset/spacenet'
preprocess_dir = '/yourdataset/spacenet_preprocess'

# image & patch
img_h, img_w = 400, 400
patch_h, patch_w = 256, 256
patch_num_h, patch_num_w = 8, 8
patch_num_h_training, patch_num_w_training = 8, 8
margin_size = 0
num_patch_infer = 256
max_node_degree = 8

# preprocessing & dataloading
is_preprocess = 0
is_multiprocess = 1
num_processes = 32
is_buffer = 0

point_mask_radius = 3
road_mask_width = 3

road_interval = 16
key_interval = 6
poly_angle_thres = 70 # degree

#vector training with teacher forcing
VECTOR_TEACHER = SimpleNamespace()
VECTOR_TEACHER.road_sample_density = 2
VECTOR_TEACHER.key_sample_density = 8
VECTOR_TEACHER.max_road_samples = 512
VECTOR_TEACHER.max_key_samples = 128
VECTOR_TEACHER.road_random_sigma = 2
VECTOR_TEACHER.key_random_sigma = 2

# Segment Inferencer
SEG_INFER = SimpleNamespace()
SEG_INFER.key_conf_thres = 0.24
SEG_INFER.road_conf_thres = 0.341
SEG_INFER.rr_nms_radius = 8
SEG_INFER.kk_nms_radius = 8
SEG_INFER.kr_nms_radius = 8

# Vector Inferencer and Settings
VECTOR_INFER = SimpleNamespace()
VECTOR_INFER.key_vector_conf_thres = 0.5
num_key_vector_anchors = 32

# Graph Inferencer and Settings
GRAPH_INFER = SimpleNamespace()
GRAPH_INFER.max_num_nbrs = 8
GRAPH_INFER.r_thres = 20
GRAPH_INFER.cos_thres = math.cos(30 / 180 * math.pi)
GRAPH_INFER.remove_cos_thres = math.cos(45 / 180 * math.pi)

#optimization
batch_size = 64
lr = 1e-4
epoch_num = 7
warmup_iter = 400

#################model########################
'''
This is the configuration of the segmentation block.
'''
# backbone
backbone = 'sam2_hiera_b+'
sam_ckpt = './sam_ckpts/sam2.1_hiera_base_plus.pt'

# neck
neck = 'fpn'
downsample_strides = [4, 8, 16, 32]
fpn_in_channel = [112, 224, 448, 896]
neck_dim = 256

# seg_head
seg_head_feat_ind = 0
seg_head = 'seg_headv2'
seg_head_in_channels = neck_dim

'''
This is the configuration of the seg2vector block.
'''
# seg encoder
seg_enc_channels = 256

#seg decoder
seg_dec_inter_channels = 256

#point decoder
point_dec_inter_channels = 256

#vector decoder
vector_dec_inter_channels = 256

##############loss########################
road_bce_loss_weight = 1
road_ftl_loss_weight = 5
road_pos_weight = 3
road_ftl_alpha = 0.3

key_bce_loss_weight = 6
key_ftl_loss_weight = 0
key_pos_weight = 1
key_ftl_alpha = 0.5

point_reg_loss_weight = 1
road_vector_loss_weight = 1
key_vector_reg_weight = 8
key_vector_cls_weight = 8
key_vector_cost_alpha = 6

########data augmentation##########
import cv2
train_augments = [
    dict(name='RandomRotate90', parameters=dict(p=1.0)),
    dict(name='HorizontalFlip', parameters=dict(p=0.5)),
    dict(name='Resize', parameters=dict(height=patch_h, width=patch_w, interpolation=cv2.INTER_CUBIC, p=1.0)),
]


###############################debug settings########################
is_debug = 0
if is_debug:
    epoch_num = 3 # debug
    warmup_iter = 1 # debug
    patch_num_h_training, patch_num_w_training = 4, 4
    patch_num_h, patch_num_w = 4, 4
    is_img_buffer = 0
    batch_size = 8
    debug_length_duplicate = 250
    train_augments = [
    dict(name='Resize', parameters=dict(height=patch_h, width=patch_w, interpolation=cv2.INTER_CUBIC, p=1.0)),
]















