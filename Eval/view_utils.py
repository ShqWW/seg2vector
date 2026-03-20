import cv2
import numpy as np

def scale_data(data, scale=4):
    data = data * scale
    return data


def scale_img(img, scale=4):
    img_scaled = cv2.resize(img, (img.shape[1]*scale, img.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    return img_scaled


def rescale_img(img, scale=4):
    img_scaled = cv2.resize(img, (img.shape[1]//scale, img.shape[0]//scale), interpolation=cv2.INTER_NEAREST)
    return img_scaled

def plot_edges(node_points, edges, edge_color=(255, 0, 0), line_width=1, img=None):
    for edge in edges:
        x1, y1 = node_points[edge[0]].astype(np.int32)
        x2, y2 = node_points[edge[1]].astype(np.int32)
        cv2.line(img, (x1, y1), (x2, y2), edge_color, line_width)  # 绘制线段
    return img

def plot_vertice(node_points, radius=5, thickness=2, point_color=(0, 180, 244), img=None):
    fill_color = (255, 255, 255)  # 白色填充
    for point in node_points:
        x, y = point.astype(np.int32)
        cv2.circle(img, (x, y), radius, point_color, -1)
        cv2.circle(img, (x, y), radius-thickness, fill_color, -1)
    return img

def plot_points(node_points, radius=5, point_color=(0, 180, 244), img=None):
    fill_color = (255, 255, 255)  # 白色填充
    for point in node_points:
        x, y = point.astype(np.int32)
        cv2.circle(img, (x, y), radius, point_color, -1)
    return img

def plot_multi_vectors(angles, angle_valids, start_points, thickness = 1, len_vector=5, angle_color=(0, 180, 244), img=None):
    '''
    angles: (N, M)
    angle_valids: (N, M)
    start_points: (N, 2)
    '''
    num_points = start_points.shape[0]
    num_angles = angles.shape[1]
    for i in range(num_points):
        start_x, start_y = start_points[i].astype(np.int32)
        for j in range(num_angles):
            if angle_valids[i, j] == 1:
                angle = angles[i, j]
                end_x = int(start_x + len_vector * np.cos(angle))
                end_y = int(start_y + len_vector * np.sin(angle))
                cv2.arrowedLine(img, (start_x, start_y), (end_x, end_y), angle_color, thickness, tipLength=0.5)
    return img

def plot_dual_vectors(angles, angle_valids, start_points, thickness = 1, len_vector=5, angle_color=(0, 180, 244), img=None):
    '''
    angles: (N, M)
    angle_valids: (N, M)
    start_points: (N, 2)
    '''
    num_points = start_points.shape[0]
    for i in range(num_points):
        start_x, start_y = start_points[i].astype(np.int32)
        for j in range(1):
            if angle_valids[i, j] == 1:
                angle = angles[i, j]
                end_x = int(start_x + len_vector * np.cos(angle))
                end_y = int(start_y + len_vector * np.sin(angle))
                cv2.arrowedLine(img, (start_x, start_y), (end_x, end_y), angle_color, thickness, tipLength=0.5)
    return img  






