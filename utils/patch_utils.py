import numpy as np

def get_patch_graph_mask(node_points, patch_info=(0, 0, 512, 512)):
    x0, y0, x1, y1 = patch_info
    patch_node_mask = (node_points[:, 0] >= x0) & (node_points[:, 0] < x1) & (node_points[:, 1] >= y0) & (node_points[:, 1] < y1)
    return patch_node_mask

def mask_patch_graph(node_points, node_property, adj_array, patch_node_mask, key_node_mask = None):
    '''
    Patch the graph based on the given patch information.
    Inputs:
    node_points: the node points, shape (N, 2) where N is the number of node points
    adj_array: the adjacency array, shape (M, 2) where M is the number of edges
    patch_node_mask: a boolean mask indicating the nodes in the patch
    Returns:
    patch_node_points, patch_adj_array
    ''' 
    patch_node_points = node_points[patch_node_mask]
    key_node_mask = key_node_mask[patch_node_mask] if key_node_mask is not None else None

    in_patch_array_mask = patch_node_mask[adj_array[:, 0]] & patch_node_mask[adj_array[:, 1]]
    patch_adj_array_with_old_ind = adj_array[in_patch_array_mask]

    index_mapping = np.zeros(len(patch_node_mask), dtype=np.int32)
    index_mapping[patch_node_mask] = np.arange(np.sum(patch_node_mask), dtype=np.int32)
    
    patch_adj_array = index_mapping[patch_adj_array_with_old_ind]
    patch_node_property = node_property[patch_node_mask]

    
    out_node_ind = patch_adj_array.flatten()
    out_degrees = np.bincount(out_node_ind, minlength=node_points.shape[0])
    
    border_node_ind = np.where(out_degrees == 1)[0]
    cross_node_ind = np.where(out_degrees > 2)[0] 

    patch_node_property[border_node_ind] = 1  # border nodes
    patch_node_property[cross_node_ind] = 3  # cross nodes

    
    return patch_node_points, patch_adj_array, patch_node_property, key_node_mask


def get_patch_cluster_mask(node_points, cluster_boundary, patch_info=(0, 0, 512, 512)):
    x0, y0, x1, y1 = patch_info
    mask = (node_points[:, 0] >= x0) & (node_points[:, 0] < x1) & (node_points[:, 1] >= y0) & (node_points[:, 1] < y1)
    patch_cluster_mask = mask[cluster_boundary[..., 0]] | mask[cluster_boundary[..., 1]]
    return patch_cluster_mask



def mask_patch_cluster(node_points, node_property, node_cluster, node_ids_in_cluster, cluster_boundary, patch_cluster_mask):
    
    patch_cluster_mask_shift = np.concatenate((np.array([False]), patch_cluster_mask), axis=0)
    node_cluster_shift = node_cluster + 1


    patch_node_cluster_mask = patch_cluster_mask_shift[node_cluster_shift]

    patch_key_node_ind = np.unique(cluster_boundary[patch_cluster_mask].flatten())
    patch_node_cluster_mask[patch_key_node_ind] = True


    patch_node_points = node_points[patch_node_cluster_mask]
    patch_node_property = node_property[patch_node_cluster_mask]
    patch_node_cluster_with_old_ind = node_cluster[patch_node_cluster_mask]
    patch_node_ids_in_cluster = node_ids_in_cluster[patch_node_cluster_mask]
    patch_cluster_boundary_with_old_ind = cluster_boundary[patch_cluster_mask]

    index_mapping = np.zeros(len(patch_cluster_mask), dtype=np.int32)
    index_mapping[patch_cluster_mask] = np.arange(np.sum(patch_cluster_mask), dtype=np.int32)
    patch_node_cluster = index_mapping[patch_node_cluster_with_old_ind] 

    index_mapping = np.zeros(len(patch_node_cluster_mask), dtype=np.int32)
    index_mapping[patch_node_cluster_mask] = np.arange(np.sum(patch_node_cluster_mask), dtype=np.int32)
    patch_cluster_boundary = index_mapping[patch_cluster_boundary_with_old_ind]

    return patch_node_points, patch_node_property, patch_node_cluster, patch_node_ids_in_cluster,patch_cluster_boundary
    
