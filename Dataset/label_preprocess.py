import os
from utils.graph_utils import *
from utils.overlap_utils import *
from utils.point_utils import *
import cv2

def graph_preprocess(label_name, out_name, label_dir, out_dir, cfg=None):
        if cfg is not None:
            dataset_name = cfg.dataset
            img_size = (cfg.img_h, cfg.img_w)
            road_interval = cfg.road_interval
            key_interval = cfg.key_interval
        else:
            print("No cfg provided!")
            return None

        node_points, adj_array = get_graph_from_json(os.path.join(label_dir, label_name), img_size)

        if dataset_name == 'spacenet':
            node_points[:, 1] = 400 - node_points[:, 1]

        overlap_points, overlap_pairs = find_overlaps(node_points, adj_array)
        node_points, adj_array = add_overlap_nodes(node_points, overlap_points, adj_array, overlap_pairs)
        node_degrees, node_poly_angles = get_node_property(node_points, adj_array)

        cluster_ids, node_ids_in_cluster, cluster_boundaries = graph2cluster(adj_array, node_degrees, node_poly_angles, poly_angle_thres=10)
        node_points, cluster_ids, node_ids_in_cluster, cluster_boundaries = add_node_to_cluster(node_points, cluster_ids, node_ids_in_cluster, cluster_boundaries, road_interval=road_interval, key_interval = key_interval)
        adj_array = cluster2graph(cluster_ids, node_ids_in_cluster, cluster_boundaries)
        
        result_dict = {
            'node_points': node_points,
            'adj_array': adj_array,
        }
        
        pickle_path = os.path.join(out_dir, out_name + '_graph.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(result_dict, f)


def mask_preprocess(label_name, out_name, label_dir, out_dir, cfg=None):
    if cfg is not None:
        dataset_name = cfg.dataset
        img_size = (cfg.img_h, cfg.img_w)
        poly_angle_thres = cfg.poly_angle_thres
        point_mask_radius = cfg.point_mask_radius
        road_mask_width = cfg.road_mask_width
    else:
        print("No cfg provided!")
        return None

    node_points, adj_array = get_graph_from_json(os.path.join(label_dir, label_name), img_size)
    if dataset_name == 'spacenet':
        node_points[:, 1] = 400 - node_points[:, 1]

    overlap_points, _ = find_overlaps(node_points, adj_array)

    node_degrees, node_poly_angles = get_node_property(node_points, adj_array)
    key_node_mask = (node_degrees >= 3) | (node_degrees == 1) | (node_poly_angles > poly_angle_thres)
    key_points = np.vstack((node_points[key_node_mask], overlap_points))

    key_mask = np.zeros(img_size, dtype=np.uint8)
    road_mask = np.zeros(img_size, dtype=np.uint8)
    overlap_mask = np.zeros(img_size, dtype=np.uint8)

    for point in key_points:
        key_mask = cv2.circle(key_mask, (int(point[0]), int(point[1])), point_mask_radius, 255, -1)
    
    for point in overlap_points:
        overlap_mask = cv2.circle(overlap_mask, (int(point[0]), int(point[1])), point_mask_radius, 255, -1)

    for edge in adj_array:
        pt1 = (int(node_points[edge[0], 0]), int(node_points[edge[0], 1]))
        pt2 = (int(node_points[edge[1], 0]), int(node_points[edge[1], 1]))
        road_mask = cv2.line(road_mask, pt1, pt2, 255, road_mask_width)

    key_path = os.path.join(out_dir, out_name + '_keypoint_mask.png')
    road_path = os.path.join(out_dir, out_name + '_road_mask.png')
    overlap_path = os.path.join(out_dir, out_name + '_overlappoint_mask.png')

    cv2.imwrite(key_path, key_mask)
    cv2.imwrite(road_path, road_mask)
    cv2.imwrite(overlap_path, overlap_mask)
    return None






    
