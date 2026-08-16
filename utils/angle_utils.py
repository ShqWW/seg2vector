import numpy as np

def get_adj_angle(node_points, adj_array, max_degree = 8):
    N = len(node_points)
    adj_array = np.concatenate((adj_array, adj_array[:, ::-1]), axis=0)
    adj_array = np.unique(adj_array, axis=0)
    sort_ind = np.argsort(adj_array[:, 0])
    adj_array = adj_array[sort_ind]

    node_degree = np.bincount(adj_array[:, 0], minlength=node_points.shape[0])

    angle_valid_mask = np.zeros((N, max_degree), dtype=bool)
    angles = np.zeros((N, max_degree), dtype=np.float32)

    row_ind = adj_array[:, 0] 
    change_ind = np.where(np.diff(row_ind) !=0)[0] + 1
    diff_ind = np.ones_like(row_ind, dtype=np.int32)
    diff_ind[0] = 0
    diff_ind[change_ind] = -(node_degree[row_ind[change_ind-1]] - 1)
    col_ind = np.cumsum(diff_ind)

    angle_valid_mask[row_ind, col_ind] = True
    vec = node_points[adj_array[:, 1]] - node_points[row_ind]
    angles[row_ind, col_ind] = np.arctan2(vec[:, 1], vec[:, 0])

    # normalize angles to [0, 2pi)
    angles[angles < 0] += 2 * np.pi
    angles[angles >= 2*np.pi] -= 2 * np.pi
    return angles, angle_valid_mask

def get_edge_angle(node_points, adj_array, edge_ind, t):
    end_points1 = node_points[adj_array[edge_ind, 0]]
    end_points2 = node_points[adj_array[edge_ind, 1]]
    vec1 = end_points1 - end_points2
    vec2 = - vec1
    angles1 = np.arctan2(vec1[:, 1], vec1[:, 0])
    angles2 = np.arctan2(vec2[:, 1], vec2[:, 0])
    angles = np.stack((angles1, angles2), axis=1)
    angles[angles < 0] += 2 * np.pi
    angles[angles >= 2*np.pi] -= 2 * np.pi
    angle_valid_mask = np.ones_like(angles, dtype=bool)
    return angles, angle_valid_mask

def angle_interp(angle_map, adj_array, array_inds, ts):
    angle_map0 = angle_map[adj_array[array_inds, 0]]
    angle_map1 = angle_map[adj_array[array_inds, 1]]
    angle_map_interp = (1 - ts)[:, np.newaxis] * angle_map0 + ts[:, np.newaxis] * angle_map1
    return angle_map_interp




