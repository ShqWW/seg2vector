import cv2
import matplotlib.pyplot as plt
import numpy as np


colors = []
for i in range(10000):
    color = np.random.randint(0, 256, 3).tolist()
    colors.append(color)

def plot_edges(node_points, edges, img_shape=(2048, 2048), edge_color=(255, 0, 0), line_width=1, img=None):
    if img is None:
        img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    for edge in edges:
        x1, y1 = node_points[edge[0]].astype(np.int32)
        x2, y2 = node_points[edge[1]].astype(np.int32)
        cv2.line(img, (x1, y1), (x2, y2), edge_color, line_width)  # 绘制线段
    return img


def plot_points(node_points, img_shape=(2048, 2048), radius=5, point_color=(0, 255, 0), img=None):
    if img is None:
        img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    for point in node_points:
        x, y = point.astype(np.int32)
        cv2.circle(img, (x, y), radius, point_color, -1)
    return img



def plot_points2(node_points, img_shape=(2048, 2048), radius=5, point_color=(0, 255, 0), thickness=-1, img=None):
    if img is None:
        img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    for point in node_points:
        x, y = point.astype(np.int32)
        cv2.circle(img, (x, y), radius, point_color, 2)
    return img

def plot_angle(node_points, angles, img_shape=(2048, 2048), ray_length = 10, point_color=(0, 255, 0), img=None):
    """
    在图像上从每个节点点按对应角度发射射线

    :param node_points: 节点点列表，每个元素为 (x, y) 坐标
    :param angles: 与节点点对应的角度（弧度）列表
    :param img_shape: 图像的形状，默认为 (2048, 2048)
    :param point_color: 射线的颜色，默认为 (0, 255, 0)
    :param img: 可选的输入图像，若为 None 则创建一个全黑图像
    :return: 绘制了射线的图像
    """
    if img is None:
        img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)

    for point, angle in zip(node_points, angles):
        x, y = point
        # 计算射线在两个方向上的终点坐标
        end_point_1_x = int(x + ray_length * np.cos(angle))
        end_point_1_y = int(y + ray_length * np.sin(angle))
        end_point_2_x = int(x - ray_length * np.cos(angle))
        end_point_2_y = int(y - ray_length * np.sin(angle))
    

        # 绘制射线
        cv2.line(img, (end_point_2_x, end_point_2_y), (end_point_1_x, end_point_1_y), point_color, 3)

    return img



def plot_road_angle(node_points, road_angles, road_angle_valid, img_shape=(2048, 2048), ray_length = 5, line_color=(255, 255, 255),line_width=1, img=None):
    if img is None:
        img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8) 


    for point, angles, angle_valid in zip(node_points, road_angles, road_angle_valid):
        x, y = point

        angles = angles[angle_valid]
        if len(angles) == 0:
            continue
        for angle in angles:
            end_point_x = int(x + ray_length * np.cos(angle))
            end_point_y = int(y + ray_length * np.sin(angle))
            cv2.line(img, (int(x), int(y)), (end_point_x, end_point_y), line_color, line_width)
            # cv2.arrowedLine(img, (int(x), int(y)), (end_point_x, end_point_y), line_color, line_width)
    return img

    


def plot_points_with_ids(node_points, ids, img_shape=(2048, 2048), radius=5, point_color=(0, 255, 0), img=None):
    if img is None:
        img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    for point, id in zip(node_points, ids):
        x, y = point.astype(np.int32)
        cv2.circle(img, (x, y), radius, point_color, -1)
        cv2.putText(img, str(int(id)), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img


def plot_points_with_ids_scale(node_points, ids, img_shape=(2048, 2048), radius=5, point_color=(0, 255, 0), img=None, scale=2):
    if img is None:
        img = np.zeros((img_shape[0]*scale, img_shape[1]*scale, 3), dtype=np.uint8)
    for point, id in zip(node_points, ids):
        x, y = (point*scale).astype(np.int32)
        cv2.circle(img, (x, y), radius, point_color, -1)
        cv2.putText(img, str(int(id)), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img

def plot_node_cluster(node_points, node_clusters, img_shape=(2048, 2048), radius=5, img=None):
    if img is None:
        img = np.zeros((*img_shape, 3), dtype=np.uint8)
    for point, cluster in zip(node_points, node_clusters):
        x, y = point.astype(int)
        cv2.circle(img, (x, y), radius, colors[cluster], -1)
    return img

def plot_node_clusters_with_ids(node_points, node_clusters, node_ids_in_cluster, img_shape=(2048, 2048), radius=5, img=None):
    if img is None:
        img = np.zeros((*img_shape, 3), dtype=np.uint8)

    for point, cluster, node_id in zip(node_points, node_clusters, node_ids_in_cluster):
        x, y = point.astype(int)
        cv2.circle(img, (x, y), radius, colors[cluster], -1)
        cv2.putText(img, str(int(node_id)), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img


def plot_edge_clusters(node_points, edges, edge_clusters, img_shape=(2048, 2048), img=None):
    if img is None:
        img = np.zeros((*img_shape, 3), dtype=np.uint8)

    for edge, cluster in zip(edges, edge_clusters):
        x1, y1 = node_points[edge[0]].astype(np.int32)
        x2, y2 = node_points[edge[1]].astype(np.int32)
        cv2.line(img, (x1, y1), (x2, y2), colors[cluster], 2)  # 绘制线段
    return img

def plot_node_spetial_edge_clusters(node_points, spetial_node2cluster, img_shape=(2048, 2048), img=None):
    if img is None:
        img = np.zeros((*img_shape, 3), dtype=np.uint8)

    # 生成不同的颜色
    edge_clusters = spetial_node2cluster[:, 1]
    unique_clusters = np.unique(edge_clusters)

    for cluster in zip(unique_clusters):
        ind = np.where(spetial_node2cluster[:, 1] == cluster)[0]
        print(ind.shape)
        node_ind0 = spetial_node2cluster[ind[0], 0]
        node_ind1 = spetial_node2cluster[ind[1], 0]
        x1, y1 = node_points[node_ind0].astype(np.int32)
        x2, y2 = node_points[node_ind1].astype(np.int32)
        cluster_index = np.where(unique_clusters == cluster)[0][0]
        cv2.line(img, (x1, y1), (x2, y2), colors[cluster_index], 2)  # 绘制线段
    return img



def plot_cluster_boundary(node_points, cluster_boundary, img_shape=(2048, 2048), img=None):
    if img is None:
        img = np.zeros((*img_shape, 3), dtype=np.uint8)

    # colors = []
    # for i in range(cluster_boundary.shape[0]):
    #     color = np.random.randint(0, 256, 3).tolist()
    #     colors.append(color) 

    for i, boundary in enumerate(cluster_boundary):
        if len(boundary) != 2:
            continue
        x1, y1 = node_points[boundary[0]].astype(np.int32)
        x2, y2 = node_points[boundary[1]].astype(np.int32)
        cv2.line(img, (x1, y1), (x2, y2), colors[i], 2)  # 绘制线段
    return img



def plot_pairs(points, pairs, links, img_shape=(2048, 2048), img=None):
    if img is None:
        img = np.zeros((*img_shape, 3), dtype=np.uint8)

    for pair, link in zip(pairs, links):
        if not link:
            continue
        ind1, ind2 = pair
        x1, y1 = points[ind1].astype(np.int32)
        x2, y2 = points[ind2].astype(np.int32)
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)  # 绘制绿色线段
    return img




def plot_pairs(points, center, nbrs, link_mask, pair_mask, img_shape=(2048, 2048), edge_color= (255, 255, 255), img=None):
    if img is None:
        img = np.zeros((*img_shape, 3), dtype=np.uint8)

    num_center = center.shape[0]
    max_nbrs = link_mask.shape[1]
    
    if points.shape[0] == 0:
        return img

    for i in range(num_center):
        center_point = points[center[i]].astype(int)
        for j in range(max_nbrs):
            if link_mask[i, j] and pair_mask[i, j]:
                neighbor_point = points[nbrs[i, j]].astype(int)
                # 绘制线段
                cv2.line(img, tuple(center_point), tuple(neighbor_point), edge_color, 1)

    return img


def plot_single_angle_map(angle_map, path='angle_map.png'):
    num_bins = angle_map.shape[-1]
    x = np.linspace(0, 360, num_bins, endpoint=False) + 180 / num_bins
    y = angle_map.flatten()
    # plot x, y
    plt.figure(figsize=(6, 6))
    plt.bar(x, y, width=360 / num_bins, align='center', edgecolor='black')
    plt.xlim(0, 360)
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Value')
    plt.title('Angle Map')
    plt.grid(True)
    plt.savefig(path)

 
