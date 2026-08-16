import rtree
import numpy as np

def find_overlap_points(segment1, segment2):
    """Calculate the intersection point of two line segments."""
    # if not do_segments_intersect(segment1, segment2):
    #     return None

    (x1, y1), (x2, y2) = segment1
    (x3, y3), (x4, y4) = segment2

    # Calculate the intersection point
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

    if 0 < t < 1 and 0 < u < 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)
    return None

def find_overlaps(node_points, adj_array):
    '''
    Find crossover points in a graph defined by node points and adjacency array.
    Inputs:
        node_points: np.ndarray of shape (N, 2) representing the coordinates of nodes
        adj_array: np.ndarray of shape (M, 2) representing pairs of indices in
        node_points that form edges
    Returns:
        crossover_points: np.ndarray of shape (K, 2) representing the coordinates of crossover
        points, where K is the number of crossover points found
    '''
    lines = np.stack((node_points[adj_array[:, 0]], node_points[adj_array[:, 1]]), axis=1)
    line_bboxes = np.sort(lines, axis=1).reshape(-1, 4)

    line_index = rtree.index.Index()
    for idx, bbox in enumerate(line_bboxes):
        line_index.insert(idx, bbox)
    overlap_points = []
    overlap_pairs = []
    tested_pairs = set()
    for i, line_0 in enumerate(lines):
        nbr_ind = list(line_index.intersection(line_bboxes[i]))
        for ni in nbr_ind:
            pair = (min(i, ni), max(i, ni))
            if pair not in tested_pairs:
                tested_pairs.add(pair)
                line_1 = lines[ni]
          
                overlap_point = find_overlap_points(line_0, line_1)
                if overlap_point is not None:
                    overlap_points.append(overlap_point)
                    overlap_pairs.append(list(pair))
    overlap_points = np.array(overlap_points, dtype=np.float32) if len(overlap_points) > 0 else np.empty((0, 2), dtype=np.float32)
    overlap_pairs = np.array(overlap_pairs, dtype=np.int32)
    return overlap_points, overlap_pairs

def add_overlap_nodes(node_points, overlap_points, adj_array, overlap_pairs):
    num_overlap = len(overlap_points)
    num_nodes = len(node_points)

    if num_overlap == 0:
        return node_points, adj_array
    
    node_points = np.vstack([node_points, overlap_points])
    
    overlap_points_ind = np.arange(num_nodes, num_nodes + num_overlap)
    overlap_adj_flatten = overlap_pairs.flatten()
    overlap_adj_unique, pairs_ind_inverse = np.unique(overlap_adj_flatten, return_inverse=True)

    overlap_points_ind = np.repeat(overlap_points_ind, 2)

    num_pairs_unique = len(overlap_adj_unique)

    adj_array_add = []


    for pair_ind in range(num_pairs_unique):
        overlap_adj = overlap_adj_unique[pair_ind]
        end_ind0 = adj_array[overlap_adj, 0]
        end_ind1 = adj_array[overlap_adj, 1]

        end_point0 = node_points[end_ind0]
        inner_points_ind = overlap_points_ind[pairs_ind_inverse == pair_ind]
        norm = np.linalg.norm(end_point0 - node_points[inner_points_ind], axis=1)
        sort_ind = np.argsort(norm)
        inner_points_ind = inner_points_ind[sort_ind]


        adj_array_add.append([end_ind0, inner_points_ind[0]])
        for i in range(len(inner_points_ind)-1):
            adj_array_add.append([inner_points_ind[i], inner_points_ind[i+1]])
        adj_array_add.append([inner_points_ind[-1], end_ind1])

    adj_array_add = np.array(adj_array_add, dtype=np.int32)
    mask = np.ones(len(adj_array), dtype=bool)
    mask[overlap_adj_unique] = False
    adj_array = adj_array[mask]
    adj_array = np.vstack([adj_array, adj_array_add])
    return node_points, adj_array
        





        