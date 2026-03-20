import torch
import numpy as np
import scipy



def key_nms_torch(points, key_points, thres):
    '''
    Using key points to supress points, all key points are kept.
    Inputs:
    points: (n, 2)
    key_points: (m, 2)
    thres: float
    Returns:
    keep_mask: (n,)
    '''
    dis_mat = torch.cdist(points, key_points)
    dis_mat_bool = dis_mat < thres
    sup_mask = torch.any(dis_mat_bool, dim=-1)
    keep_mask = ~sup_mask
    return keep_mask


def key_nms(points, key_points, thres):
    '''
    Using key points to supress points, all key points are kept.
    Inputs:
    points: (n, 2)
    key_points: (m, 2)
    thres: float
    Returns:
    keep_mask: (n,)
    '''
    tree = scipy.spatial.cKDTree(key_points)
    distances, _ = tree.query(points)
    sup_mask = distances < thres
    keep_mask = ~sup_mask
    return keep_mask


def point_nms_torch(points, points_scores, thres):
    '''
    Cluster NMS for points set in pytorch version, it's faster than traditional NMS
    Inputs:
    points: (n, 2)
    points_scores: (n,)
    thres: float
    Return:
    nms_points: (k, 2)  k<=n
    '''
    sort_ind = torch.argsort(points_scores, descending=True)
    points = points[sort_ind]

    dis_mat = torch.cdist(points, points)
    mask = torch.tril(torch.ones_like(dis_mat, dtype=bool), diagonal=-1)
    dis_mat_bool = torch.logical_and(dis_mat < thres, mask)

    sup_mask = torch.zeros((points.shape[0]), dtype=bool).to(points.device)
    changed = True  

    while changed:
        new_sup_mask = torch.any(dis_mat_bool & ~sup_mask[None, :], dim=-1)
        changed = not torch.all(new_sup_mask == sup_mask)
        sup_mask = new_sup_mask

    nms_points = points[~sup_mask]


    # dis_mat = torch.norm(nms_points[:, None, :] - nms_points[None, :, :], dim=-1)
    # mask = torch.tril(torch.ones_like(dis_mat, dtype=bool), diagonal=-1)
    # dis_mat[~mask] = 1e6
    # print(torch.min(dis_mat))

    return nms_points


def point_nms(points, scores, thres):
    '''
    Cluster NMS for points set in pytorch version, it's faster than traditional NMS
    Inputs:
    points: (n, 2)
    points_scores: (n,)
    thres: float
    Return:
    nms_points: (k, 2)  k<=n
    '''
    # if score > 1.0, the point is forced to be kept regardless

    sorted_indices = np.argsort(scores)[::-1]
    sorted_points = points[sorted_indices, :]
    sorted_scores = scores[sorted_indices]
    kept = np.ones(sorted_indices.shape[0], dtype=bool)
    tree = scipy.spatial.cKDTree(sorted_points)
    for idx, p in enumerate(sorted_points):
        if not kept[idx]:
            continue
        # neighbor_indices = tree.query_radius(p[np.newaxis, :], r=radius)[0]
        neighbor_indices = tree.query_ball_point(p, r=thres)
        neighbor_indices = np.array(neighbor_indices)
        neighbor_scores = sorted_scores[neighbor_indices]
        remove_nbr = (neighbor_scores <= sorted_scores[idx])
        remove_indices = neighbor_indices[remove_nbr] 
        kept[remove_indices] = False
        kept[idx] = True
    
    nms_points = sorted_points[kept]
    return nms_points


# def point_nms(points, scores, thres):
#     points = torch.from_numpy(points)
#     scores = torch.from_numpy(scores)
#     nms_points = point_nms_torch(points, scores, thres)
#     return nms_points.numpy()



def point_nms_torch_hd(points, points_scores, thres):
    '''
    nms points of the single batch
    key_points: (m, 2)
    key_points_scores: (m,)
    '''
        # Sort key points by their scores in descending order
    sorted_scores, indices = torch.sort(points_scores, descending=True)
    sorted_key_points = points[indices]
    
    # Initialize the list of indices to keep
    keep_indices = []
    
    while sorted_key_points.size(0) > 0:
        # Keep the key point with the highest score
        keep_indices.append(indices[0].item())
        
        # Compute Euclidean distance between the first key point and the rest
        if sorted_key_points.size(0) == 1:
            break
            
        distances = torch.norm(sorted_key_points[1:] - sorted_key_points[0], dim=1)
        
        # Remove key points with distance less than the threshold
        mask = distances >= thres
        sorted_key_points = sorted_key_points[1:][mask]
        indices = indices[1:][mask]
    keep_mask= torch.tensor(keep_indices, dtype=torch.long)

    points = points[keep_mask]

    return points, keep_mask   