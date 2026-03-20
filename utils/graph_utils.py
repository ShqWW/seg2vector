from networkx import node_degree_xy
import numpy as np
import pickle
import math
from scipy.spatial import cKDTree
from scipy.spatial import KDTree

def filter_duplicate_edges(adj_array):
    '''
    Remove duplicate edges of graph.
    Inputs:
    adj_array: the adjacency array, shape (num_edges, 2)

    Returns:
    adj_array: the adjacency array after removing duplicate edges, shape (num_edges, 2
    ''' 
    adj_array = np.sort(adj_array, axis=1)
    adj_array = np.unique(adj_array, axis=0)
    return adj_array


def get_graph_from_json(json_file, img_size=(2048, 2048)):
    '''
    Read the graph from a JSON file and convert it to a vertex points and adjacency array.
    The JSON file should contain a dictionary where the keys are the coordinates of the vertices and the values are lists of adjacent vertices.
    Inputs:
    json_file: the path to the JSON file
    Returns:
    node_points: the xy coordinates of the vertices, shape (num_nodes, 2) 
    adj_array: the adjacency array, shape (num_edges, 2), the adjacency array, shape (num_edges, 2), each row contains the indices of the two nodes that form an edge(not directed edges).
    '''
    json_graph = pickle.load(open(json_file, 'rb'))
    points = []
    for n, neis in json_graph.items():
        for nei in neis:
            points.append([int(n[1]), int(n[0]), int(nei[1]), int(nei[0])])
    points = np.array(points).reshape(-1, 2, 2)

    bound_mask = (np.max(points[..., 0], axis=-1) < img_size[1]) & (np.max(points[..., 1], axis=-1) < img_size[0]) & (np.min(points.reshape(-1, 4), axis=-1) >= 0)
    points = points[bound_mask]

    all_points = points.reshape(-1, 2)
    node_points, point_indices = np.unique(all_points, axis=0, return_inverse=True)
    adj_array = point_indices.reshape(-1, 2)

    node_points = node_points.astype(np.float32)
    adj_array = adj_array.astype(np.int32)

    # Since the edges is not directed edges, the duplicate edges can be removed
    adj_array = filter_duplicate_edges(adj_array)
    return node_points, adj_array

def transfer_graph_to_dict(node_points, adj_array):
    '''
    Convert the vertex points and adjacency array to a graph in JSON format.
    Inputs:
    node_points: the xy coordinates of the vertices, shape (num_nodes, 2)
    adj_array: the adjacency array, shape (num_edges, 2), each row contains the indices of the two vertices that form an edge (not directed edges).
    Returns:
    json_graph: a dictionary where the keys are the coordinates of the vertices and the values are lists of adjacent vertices.
    The coordinates are in the format (y, x) to match the image coordinate system.
    The adjacency array is undirected, so each edge is represented twice in the JSON graph.
    '''
    graph_dict = {}

    # double the edges to make it undirected
    adj_array = np.concatenate((adj_array, adj_array[:, ::-1]), axis=0)
    adj_array = np.unique(adj_array, axis=0)  

    for edge in adj_array:
        index1, index2 = int(edge[0]), int(edge[1])
        point1 = tuple(float(x) for x in node_points[index1])
        point2 = tuple(float(x) for x in node_points[index2])
        key1 = (point1[1], point1[0])
        key2 = (point2[1], point2[0])
        if key1 not in graph_dict:
            graph_dict[key1] = []
        if key2 not in graph_dict[key1]:
            graph_dict[key1].append(key2)
    return graph_dict


def add_nodes(node_points, adj_array, resolution=8):
    num_edge = len(adj_array)
    num_points = len(node_points)
    end_points1 = node_points[adj_array[:, 0]]
    end_points2 = node_points[adj_array[:, 1]]
    edge_vectors = end_points2 - end_points1
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)

    new_edges_list = []
    new_points_list = [] 
    for edge_no in range(num_edge):
        end_no1, end_no2 = adj_array[edge_no]
        edge_length = edge_lengths[edge_no]
        sample_count = max(0, int(edge_length / resolution) - 1)

        if sample_count == 0:
            new_edges_list.append(np.array([[end_no1, end_no2]], dtype=np.int32))
            continue

        end_point1, edge_vector = end_points1[edge_no], edge_vectors[edge_no]
        samples = np.linspace(0.0, 1.0, sample_count + 2, endpoint=True)
        sample_ids = np.arange(sample_count) + num_points
        sample_pts = end_point1[np.newaxis, :] + samples[:, np.newaxis] * edge_vector[np.newaxis, :]

        new_node_points = sample_pts[1:-1, :]
        new_points_list.append(new_node_points)
        
        new_edges = np.column_stack([np.concatenate(([end_no1], sample_ids)), np.concatenate((sample_ids, [end_no2]))])
        new_edges_list.append(new_edges.astype(np.int32))

        num_points += sample_count
    
    new_points = np.vstack(new_points_list) if len(new_points_list) > 0 else np.zeros((0, 2), dtype=np.float32)
    new_edges = np.vstack(new_edges_list) if len(new_edges_list) > 0 else np.zeros((0, 2), dtype=np.int32)

    node_points = np.vstack([node_points, new_points])
    return node_points, new_edges

def get_node_property(node_points, adj_array, filter_duplicate=True):
    '''
    Inputs:
    node_points: the coordinates of the nodes, shape (num_nodes, 2)
    adj_array: the adjacency array, shape (num_edges, 2), each row contains the indices of the two vertices that form an directed edge from the first col to the second col
    Returns:
    node_degrees: the degree of each node, shape (num_nodes,)
    node_poly_angle: the angle of each two-node intersection node, shape (num_nodes,), -1 for non-two-node intersection nodes, range [0, 180) 
    '''

    if filter_duplicate:
        adj_array = filter_duplicate_edges(adj_array)

    adj_array_flatten = adj_array.flatten()
    inv_adj_array_flatten = adj_array[:, ::-1].flatten()
    node_degrees = np.bincount(adj_array_flatten, minlength=node_points.shape[0])

    deg2_ind = np.where(node_degrees == 2)[0]
    node_poly_angles = - np.ones(node_points.shape[0], dtype=np.float32)

    
    node_mask = adj_array_flatten[:, np.newaxis] == deg2_ind

    edge_indices = np.where(node_mask.T)[1].reshape(-1, 2)

    # get the two adjacent nodes for each edge
    deg2_ind_nbr1 = inv_adj_array_flatten[edge_indices[:, 0]]
    deg2_ind_nbr2 = inv_adj_array_flatten[edge_indices[:, 1]]

    # get the vectors from the two adjacent nodes to the two-node intersection node
    vec1 = node_points[deg2_ind_nbr1] - node_points[deg2_ind]
    vec2 = node_points[deg2_ind_nbr2] - node_points[deg2_ind]

    cos_angle = - np.sum(vec1 * vec2, axis=1) / (np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1)) # the minus sign means the angle = 180 - original_angle
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle) * 180 / math.pi

    node_poly_angles[deg2_ind] = angle
    return node_degrees, node_poly_angles


def get_key_node_mask(node_points, adj_array, poly_angle_thres = 180):
    if len(node_points) == 0:
        return np.array([], dtype=bool)
    node_degrees, node_poly_angles = get_node_property(node_points, adj_array)
    key_mask = (node_degrees >= 3) | (node_degrees == 1) | (node_poly_angles > poly_angle_thres)
    return key_mask


def node_in_patch(node_points, patch_h, patch_w):
    if len(node_points) == 0:
        return np.array([], dtype=bool)
    node_mask = (node_points[:, 0] >= 0) & (node_points[:, 0] < patch_w) & (node_points[:, 1] >= 0) & (node_points[:, 1] < patch_h)
    return node_mask


def cal_distance(points1, points2):
    '''
    Inputs:
    points1: shape (N, 2), where N is the number of points1
    points2: shape (M, 2), where M is the number of points2
    Returns:
    dis_mat: shape (N, M), where dis_mat[i, j] is the distance between points1[i] and points2[j]
    '''
    dis_mat = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]
    dis_mat = np.linalg.norm(dis_mat, axis=-1)
    return dis_mat




def graph2cluster(adj_array, node_degrees, node_poly_angles, poly_angle_thres = 10):
    '''
    Convert the representation of road from graph to cluster. Compard with graph representation, the cluster representation is convenient for further processing, such as add or remove nodes along the node points. 
    Each cluster is a connected component of nodes, which denotes a single road segment without intersections.

    Inputs:
    node_points: the coordinates of the nodes, shape (num_nodes, 2)
    adj_array: the adjacency array, shape (num_edges, 2), each row contains the indices of the two vertices that form an directed edge from first col to second col.
    Returns:
    cluster_ids: the cluster id of each node, shape (num_nodes,), where each value
    is the id of the cluster that the node belongs to, -1 if the node is a boundary node.
    node_ids_in_cluster: the id of the node in the cluster, shape (num_nodes,), where each value is the id of the node in the cluster, -1 if the node is a boundary node.
    cluster_boundary: the boundary of each cluster, shape (num_clusters, 2), where each row contains the indices of the start and end node indices that form the boundary of the cluster.
    
    ''' 
    num_nodes = len(np.unique(adj_array))
    # construct the adjacency list
    adj_list = [[] for _ in range(num_nodes)]
    for u, v in adj_array:
        adj_list[u].append(v)
        adj_list[v].append(u)

    cluster_ids = - np.ones(num_nodes, dtype=int)
    node_ids_in_cluster = - np.ones(num_nodes, dtype=np.int32)

    cluster_id = 0
    # visited array to keep track of visited nodes
    visited = np.zeros(num_nodes, dtype=bool)
    # define the end node mask
    cluster_end_mask = (node_degrees==1) | (node_degrees>2) | (node_poly_angles>=poly_angle_thres)

    # DFS to find connected components
    cluster_boundary_list = []

    for end_ind in np.where(cluster_end_mask)[0]:
        visited[end_ind] = True
        for cluster_nbr in adj_list[end_ind]:
            # if the node is not visited
            if not visited[cluster_nbr]:
                next_node = cluster_nbr
                current_node = end_ind
                node_id_in_cluster = 1
                # DFS until reaching a end node
                while not cluster_end_mask[next_node]:
                    last_node = current_node
                    current_node = next_node
                    visited[current_node] = True
                    cluster_ids[current_node] = cluster_id
                    node_ids_in_cluster[current_node] = node_id_in_cluster
                    for nbr in adj_list[current_node]:
                        if nbr != last_node:
                            next_node = nbr
                    node_id_in_cluster += 1


                cluster_boundary_list.append([end_ind, next_node])
                cluster_id += 1
    cluster_boundaries = np.array(cluster_boundary_list, dtype=np.int32)
    return cluster_ids, node_ids_in_cluster, cluster_boundaries

def add_node_to_cluster(node_points, cluster_ids, node_ids_in_cluster, cluster_boundaries, road_interval=4, key_interval=6):
    '''
    Add nodes to the cluster by sampling points along the edges of the cluster boundary to improve the resolution of the road.
    Inputs:
    node_points: the coordinates of the nodes, shape (num_nodes, 2)
    cluster_ids: the cluster id of each node, shape (num_nodes,)
    node_ids_in_cluster: the id of the node in the cluster, shape (num_nodes,)
    cluster_boundary: the boundary of each cluster, shape (num_clusters, 2)
    road interval: the interval for sampling points along the road, in pixel unit. The smaller the interval, the more points will be added.
    key_interval: the interval for the end points and inter points
    Returns:
    '''
    num_cluster = len(cluster_boundaries)
    boundary_node_idx, inv_idx = np.unique(cluster_boundaries.flatten(), return_inverse=True)
    new_cluster_boudaries = np.arange(boundary_node_idx.shape[0])[inv_idx].reshape(cluster_boundaries.shape)
    boundary_node_points = node_points[boundary_node_idx]

    new_points_list = []
    new_cluster_ids_list = []
    new_node_ids_in_cluster_list = []

    for cluster_id in range(num_cluster):
        cluster_endpoints = boundary_node_points[new_cluster_boudaries[cluster_id]]

        cluster_node_idx = np.where(cluster_ids == cluster_id)[0]
        node_ids = node_ids_in_cluster[cluster_node_idx]

        # sort the inner nodes by their ids in cluster
        sorted_indices = np.argsort(node_ids)

        # get the sorted results
        node_ids = node_ids[sorted_indices]
        cluster_node_idx = cluster_node_idx[sorted_indices]
        cluster_inner_points = node_points[cluster_node_idx]


        cluster_sequence = np.vstack([cluster_endpoints[0], cluster_inner_points, cluster_endpoints[1]])
        cluster_length = cluster_sequence.shape[0]

        sequence_diff = cluster_sequence[1:] - cluster_sequence[:-1]
        lengths = np.linalg.norm(sequence_diff, axis=1)
        cumsum_lengths = np.insert(np.cumsum(lengths), 0, 0) # add zero at the beginning
        total_length = cumsum_lengths[-1]
        t =  cumsum_lengths/total_length
        
        if total_length <= key_interval:
            new_t = np.array([], dtype=np.float32)
        elif total_length <= 2 * key_interval:
            new_t = np.array([0.5], dtype=np.float32)
        else:
            t_start = key_interval / total_length
            t_end = 1 - t_start
            num_add_points = int((t_end - t_start) * total_length / road_interval) + 2
            new_t = np.linspace(t_start, t_end, num_add_points, endpoint=True)

        add_points = np.vstack((np.interp(new_t, t, cluster_sequence[..., 0]), np.interp(new_t, t, cluster_sequence[..., 1]))).transpose()  if new_t.shape[0] >0 else np.array([]).reshape(0, 2)

        new_points_list.append(add_points)
        new_cluster_ids_list.append(cluster_id * np.ones(add_points.shape[0], dtype=np.int32))
        new_node_ids_in_cluster_list.append(np.arange(len(add_points), dtype=np.int32))

    new_node_points = np.vstack(new_points_list)
    new_cluster_ids = np.concatenate(new_cluster_ids_list)
    new_node_ids_in_cluster = np.concatenate(new_node_ids_in_cluster_list)

    node_points = np.vstack([boundary_node_points, new_node_points])
    cluster_ids = np.concatenate([ -1 * np.ones(len(boundary_node_idx), dtype=np.int32), new_cluster_ids])
    node_ids_in_cluster = np.concatenate([ -1 * np.ones(len(boundary_node_idx), dtype=np.int32), new_node_ids_in_cluster])
    cluster_boundaries = new_cluster_boudaries

    return node_points, cluster_ids, node_ids_in_cluster, cluster_boundaries

def cluster2graph(cluster_ids, node_ids_in_cluster, cluster_boundaries):
    '''
    Convert the cluster representation back to the garph representation.
    Inputs:
    node_cluster_idscluster: the cluster id of each node, shape (num_nodes,), where each value
    is the id of the cluster that the node belongs to, -1 if the node is
    not in any cluster
    node_ids_in_cluster: the id of the node in the cluster, shape (num_nodes,), where each value is the id of the node in the cluster, -1 if the node is a cross node (belong to multiple clusters or just an end node denotes the end of the road)
    cluster_boundaries: the boundary of each cluster, shape (num_clusters, 2), where
    each row contains the indices of the two vertices that form the boundary of the cluster.
    Returns:
    adj_array: the adjacency array, shape (num_edges, 2), each row contains
    the indices of the two vertices that form an directed edge from first col to second col.
    '''
    num_cluster = len(cluster_boundaries)
    # find the node indices for each cluster
    cluster_indices = [np.where(cluster_ids == i)[0] for i in range(num_cluster)]
    # init the adj list
    adj_list = []
    # processing each cluster
    for i, inner_node in enumerate(cluster_indices):
        if len(inner_node) == 0:
            adj_list.append(cluster_boundaries[i])
            continue
        inner_node_ids = node_ids_in_cluster[inner_node]
        
        sorted_indices = np.argsort(inner_node_ids)
        sort_inner_node = inner_node[sorted_indices]

        start_edge = np.array([cluster_boundaries[i, 0], sort_inner_node[0]], dtype=np.int32)
        inner_edges = np.column_stack([sort_inner_node[:-1], sort_inner_node[1:]])
        end_edge = np.array([sort_inner_node[-1], cluster_boundaries[i, 1]], dtype=np.int32)
        adj_list.extend([start_edge, inner_edges, end_edge])
    # convert the adjacency list to the overall adjacency array
    adj_array = np.vstack(adj_list)
    adj_array = filter_duplicate_edges(adj_array)
    return adj_array



def mask_graph(node_mask, adj_array, *nodes_tuple):
    '''
    mask the graph by the node mask.
    Inputs:
    node_mask: a boolean array of shape (num_nodes,), where True indicates the node is
    kept and False indicates the node is removed.
    adj_array: the adjacency array, shape (num_edges, 2)
    nodes_tuple: a tuple of node arrays to be masked, each array has shape (num_nodes, D) or (num_nodes,)
    '''
    edge_mask = node_mask[adj_array[:, 0]] & node_mask[adj_array[:, 1]]
    mask_adj_array_with_old_ind = adj_array[edge_mask]
    index_mapping = np.zeros(len(node_mask), dtype=np.int32)
    index_mapping[node_mask] = np.arange(np.sum(node_mask), dtype=np.int32)
    mask_adj_array = index_mapping[mask_adj_array_with_old_ind]

    mask_nodes_list = []
    for nodes in nodes_tuple:
        mask_nodes = nodes[node_mask]
        mask_nodes_list.append(mask_nodes)
    nodes_tuple = tuple(mask_nodes_list)
    return mask_adj_array, *nodes_tuple

    

















     









    


    
 




