import numpy as np
from scipy.spatial import cKDTree
from .graph_utils import get_node_property
import torch


def get_points_on_edges(node_points, adj_array, edge_inds, t):
    choise_edges = adj_array[edge_inds]
    start_points = node_points[choise_edges[:, 0]]
    end_points = node_points[choise_edges[:, 1]]
    sampled_points = start_points + (end_points - start_points) * t[:, np.newaxis]
    return sampled_points   

def point_segment_dist(points, line_seg):
    '''
    calculate the polar coordinates of the line segments
    points: shape (N, 2) N is the number of points, 2 is the x and y coordinates.
    line_seg: shape (N, 2, 2) the first 2 is the two end points, the second 2 is the x and y coordinates. the line segments must be in the cartesian coordinates with corresponding origins.
    Return:
    
    '''
    AB = line_seg[:, 1, :] - line_seg[:, 0, :]  
    AO = -line_seg[:, 0, :] + points
    AB_squared_norm = np.sum(AB ** 2, axis=1, keepdims=True)
    t = np.sum(AO * AB, axis=1, keepdims=True) / AB_squared_norm
    rhos_vector = -AO + t * AB
    t = t.reshape(-1)
    rhos = np.linalg.norm(rhos_vector, axis=1)
    valid_mask1 = t < 0
    valid_mask2 = t > 1
    rhos[valid_mask1] = np.linalg.norm(points[valid_mask1] - line_seg[valid_mask1, 0, :], axis=-1)
    rhos[valid_mask2] = np.linalg.norm(points[valid_mask2] - line_seg[valid_mask2, 1, :], axis=-1)
    return rhos, t

def match_points_to_graph(points, node_points, adj_array, dist_thres=5, k=5):
    N = points.shape[0]
    k = min(k, len(adj_array))
    p0 = node_points[adj_array[:, 0]]
    p1 = node_points[adj_array[:, 1]]
    mid_points = (p0 + p1) / 2.0
    tree = cKDTree(mid_points)
    _, edge_inds = tree.query(points, k=k)
    
    if k==1:
        edge_inds = edge_inds[:, np.newaxis]

    points_expand = np.repeat(points, k, axis=0)
    edge_inds_expand = edge_inds.reshape(-1)


    line_segs = np.stack((node_points[adj_array[edge_inds_expand, 0]],
                            node_points[adj_array[edge_inds_expand, 1]]), axis=1)

    dists, ts = point_segment_dist(points_expand, line_segs)
    
    dists = dists.reshape(N, k)
    ts = ts.reshape(N, k)

    min_ind = np.argmin(dists, axis=1)
    ts = np.clip(ts, 0.0, 1.0)

    matched_edge_inds = edge_inds[np.arange(N), min_ind]
    matched_ts = ts[np.arange(N), min_ind]
    matched_dists = dists[np.arange(N), min_ind]
    match_valids = matched_dists <= dist_thres

    matched_edges = adj_array[matched_edge_inds]
    p0_matched = node_points[matched_edges[:, 0]]
    p1_matched = node_points[matched_edges[:, 1]]
    target_points = p0_matched + (p1_matched - p0_matched) * matched_ts[:, np.newaxis]
    return target_points, matched_edge_inds, matched_ts, match_valids, matched_dists



def match_points_to_points(points, target_points):
    # find the nearest point in target_points index for each point in points
    tree =  cKDTree(target_points)
    dists, inds = tree.query(points, k=1)
    return dists, inds


def sample_t_on_graph_with_density(node_points, adj_array, density=10.0):
    edge_length = np.linalg.norm(node_points[adj_array[:, 0]] - node_points[adj_array[:, 1]], axis=1)
    total_length = np.sum(edge_length)
    num_bin = int(total_length // density)
    real_density = total_length / num_bin
    bin_starts = np.linspace(0, total_length, num=num_bin, endpoint=False, dtype=None, axis=0)
    edge_boundaries = np.insert(np.cumsum(edge_length), 0, 0)
    offset = np.random.rand((num_bin)) * real_density
    sample_positions = bin_starts + offset
    sample_inds = np.digitize(sample_positions, edge_boundaries) - 1
    sample_inds = np.clip(sample_inds, 0, len(edge_length)-1)
    t = (sample_positions - edge_boundaries[sample_inds]) / edge_length[sample_inds]
    return sample_inds, t


def sample_road_points_on_patch(node_points, key_node_mask, adj_array, density = 10, patch_h=512, patch_w=512, max_sample=512, random_sigma=0, key_thres = 5.0):
    if len(node_points) ==0:
        sample_points = np.zeros((max_sample, 2), dtype=np.float32)
        sample_valid = np.zeros((max_sample,), dtype=np.bool_)
        sample_inds = None
        t = None
    else:
        sample_inds, t = sample_t_on_graph_with_density(node_points, adj_array, density=density)
        sample_points = get_points_on_edges(node_points, adj_array, sample_inds, t)
        if random_sigma > 0:
            noise = (np.random.randn(*sample_points.shape).astype(np.float32)) * random_sigma
            sample_points += noise
        mask = (sample_points[:, 0] >=0) & (sample_points[:, 0] < patch_w) & (sample_points[:, 1] >=0) & (sample_points[:, 1] < patch_h)
        if random_sigma > 0:
            _, sample_inds, t, _, _ = match_points_to_graph(sample_points, node_points, adj_array, k=5)
        
        key_node_points = node_points[key_node_mask]
        if len(key_node_points) > 0:
            dists, _ = match_points_to_points(sample_points, key_node_points)
            mask = mask & (dists >= key_thres)

        sample_points = sample_points[mask]
        sample_inds = sample_inds[mask]
        t = t[mask]
        sample_valid = np.ones((sample_points.shape[0],), dtype=np.bool_)

        if len(sample_points) > max_sample:
            select_inds = np.random.choice(len(sample_points), size=(max_sample,), replace=False)
            sample_points = sample_points[select_inds]
            sample_inds = sample_inds[select_inds]
            t = t[select_inds]
            sample_valid = sample_valid[select_inds]
        elif len(sample_points) < max_sample:
            pad_num = max_sample - len(sample_points)
            sample_points = np.vstack((sample_points, np.zeros((pad_num, 2), dtype=np.float32)))
            sample_inds = np.hstack((sample_inds, np.zeros((pad_num,), dtype=np.int32)))
            t = np.hstack((t, np.zeros((pad_num,), dtype=np.float32)))
            sample_valid = np.hstack((sample_valid, np.zeros((pad_num,), dtype=np.bool_)))
    return sample_points, sample_inds, t, sample_valid



def sample_key_points_on_patch(key_points, density = 3, patch_h=512, patch_w=512, max_sample=64, random_sigma=0):
    num_key = len(key_points)
    if num_key ==0:
        sample_points = np.zeros((max_sample, 2), dtype=np.float32)
        sample_valid = np.zeros((max_sample,), dtype=np.bool_)
        sample_inds = None
    else:
        sample_inds = np.repeat(np.arange(num_key), density)
        sample_points = key_points[sample_inds]
        if random_sigma > 0:
            noise = np.random.rand(*sample_points.shape).astype(np.float32) * random_sigma
            sample_points += noise
        mask = (sample_points[:, 0] >=0) & (sample_points[:, 0] < patch_w) & (sample_points[:, 1] >=0) & (sample_points[:, 1] < patch_h)
        if random_sigma > 0:
           _, sample_inds = match_points_to_points(sample_points, key_points)

        sample_points = sample_points[mask]
        sample_inds = sample_inds[mask]
        sample_valid = np.ones((sample_points.shape[0],), dtype=np.bool_)


        permuted_idx = np.random.permutation(sample_points.shape[0])
        sample_points = sample_points[permuted_idx]
        sample_inds = sample_inds[permuted_idx]
        sample_valid = sample_valid[permuted_idx]

        #

        if len(sample_points) > max_sample:
            select_inds = np.random.choice(len(sample_points), size=(max_sample,), replace=False)
            sample_points = sample_points[select_inds]
            sample_inds = sample_inds[select_inds]
            sample_valid = sample_valid[select_inds]
        elif len(sample_points) < max_sample:
            pad_num = max_sample - len(sample_points)
            sample_points = np.vstack((sample_points, np.zeros((pad_num, 2), dtype=np.float32)))
            sample_inds = np.hstack((sample_inds, np.zeros((pad_num,), dtype=np.int32)))
            sample_valid = np.hstack((sample_valid, np.zeros((pad_num,), dtype=np.bool_)))
    return sample_points, sample_inds, sample_valid






def padding_tensor_torch(points, target_length, content = -1, dim=0):
    """
    将输入的 points 填充到指定的长度，使用 -1 进行填充
    :param points: 输入的点张量，可以是一维或多维，最后一个维度为要填充的维度
    :param target_length: 目标填充长度
    :return: 填充后的张量
    """
    current_length = points.size(dim)
    if current_length >= target_length:
        return points
    padding_length = target_length - current_length
    # 构建填充张量，形状除了最后一维与原张量相同，最后一维为 padding 长度
    padding_shape = list(points.size())
    padding_shape[dim] = padding_length
    padding = torch.full(padding_shape, content, dtype=points.dtype, device=points.device)
    # 拼接原张量和填充张量
    result = torch.cat((points, padding), dim=dim)
    return result

    


    
# def sample_edges(node_points, adj_array, num_samples=512):
#     edge_inds = len(adj_array)
#     edge_length = np.linalg.norm(node_points[adj_array[:, 0]] - node_points[adj_array[:, 1]], axis=1)
#     edge_probs = edge_length / np.sum(edge_length)
#     sample_inds = np.random.choice(edge_inds, size=num_samples, replace=True, p=edge_probs)
#     return sample_inds

# def get_sample_t(num_samples=512):
#     t = np.random.rand(num_samples).astype(np.float32)
#     return t


def get_center_grid(img_size=(2048, 2048), down_sampling_factor=8):
    '''
    Generate a polar grid image with the specified size and downsampling factor.
    Inputs:
    img_size: the size of the image, default is (2048, 2048
    down_sampling_factor: the factor to downsample the image, default is 8
    Returns:
    center_grid: the local pole grid, shape (H, W, 2) where H and W are the height and width of the downsampled image
    '''
    down_sampling_x, down_sampling_y = img_size[1] // down_sampling_factor, img_size[0] // down_sampling_factor
    grid_x, grid_y = np.meshgrid(np.arange(0, down_sampling_x, 1, dtype=np.float32), np.arange(0, down_sampling_y, 1, dtype = np.float32))
    grid_x, grid_y = grid_x*down_sampling_factor + down_sampling_factor//2 - 0.5, grid_y*down_sampling_factor + down_sampling_factor//2 - 0.5
    center_grid = np.stack((grid_x, grid_y), axis=-1)
    return center_grid


def get_road_grid_map(node_points, adj_array, img_size=(512, 512), down_factor=8, distance_thres=16):
    if len(node_points)==0 and len(adj_array)==0:
        offset_map = np.zeros((img_size[1]//down_factor, img_size[0]//down_factor, 2), dtype=np.float32)
        conf_map = np.zeros((img_size[1]//down_factor, img_size[0]//down_factor), dtype=np.float32)
        return offset_map, conf_map
    center_grid = get_center_grid(img_size=img_size, down_sampling_factor=down_factor)
    center_grid = center_grid.reshape(-1, 2)
    target_points, _, _, _, dists = match_points_to_graph(center_grid, node_points, adj_array, k=5)

    
    conf_map = (dists <= distance_thres).astype(np.float32)
    offset_map = target_points - center_grid
    offset_map = offset_map.reshape((img_size[1]//down_factor, img_size[0]//down_factor, 2))
    conf_map = conf_map.reshape((img_size[1]//down_factor, img_size[0]//down_factor))
    return offset_map, conf_map


def get_key_grid_map(key_points, img_size=(512, 512), down_factor=8, distance_thres=16):
    if len(key_points)==0:
        offset_map = np.zeros((img_size[1]//down_factor, img_size[0]//down_factor, 2), dtype=np.float32)
        conf_map = np.zeros((img_size[1]//down_factor, img_size[0]//down_factor), dtype=np.float32)
        return offset_map, conf_map
    center_grid = get_center_grid(img_size=img_size, down_sampling_factor=down_factor)
    center_grid = center_grid.reshape(-1, 2)
    dists, inds = match_points_to_points(center_grid, key_points)
    target_points = key_points[inds]
    conf_map = (dists <= distance_thres).astype(np.float32)
    offset_map = target_points - center_grid
    conf_map = conf_map.reshape((img_size[1]//down_factor, img_size[0]//down_factor))
    offset_map = offset_map.reshape((img_size[1]//down_factor, img_size[0]//down_factor, 2))
    return offset_map, conf_map


def proposal_point_from_map(proposal_map, down_factor=8):
    map_h, map_w = proposal_map.shape[1], proposal_map.shape[2]
    center_grid = get_center_grid((map_h*down_factor, map_w*down_factor), down_factor)
    diff_map = proposal_map[1:, :, :].transpose(1, 2, 0).reshape(-1, 2)
    conf_map = proposal_map[0, :, :].reshape(-1)
    proposal_points = center_grid.reshape(-1, 2) + diff_map
    valid_mask = conf_map > 0.5
    proposal_points = proposal_points[valid_mask]
    return proposal_points







    








    
