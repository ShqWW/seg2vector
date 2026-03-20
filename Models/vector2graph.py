import torch
from torch import nn
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

class Vector2Graph(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.graph_infer = cfg.GRAPH_INFER
        self.img_h, self.img_w = cfg.img_h, cfg.img_w

    def forward(self, key_points, road_points, key_angles, road_angles, key_angle_valids, road_angle_valids):
        node_points, angles, angle_valids, key_node_mask = self.combine_key_road(road_points, key_points, road_angles, key_angles, road_angle_valids, key_angle_valids)
        nbr_inds = self.get_node_nbr(node_points)
        assign_mask = torch.zeros_like(nbr_inds, dtype=torch.bool)

        assign_mask[key_node_mask] = self.dual_connection(node_points, angles, angle_valids, nbr_inds, key_node_mask)
        angles, angle_valids = self.remove_angles(node_points, angles, angle_valids, nbr_inds, assign_mask)
        assign_mask[~key_node_mask] = self.dual_connection(node_points, angles, angle_valids, nbr_inds, ~key_node_mask)

        ind1, ind2 = torch.nonzero(assign_mask, as_tuple=True)
        adj_array = torch.stack((ind1, nbr_inds[ind1, ind2]), dim=-1)
        adj_array = torch.sort(adj_array, dim=1)[0]
        adj_array = torch.unique(adj_array, dim=0)
        return node_points, adj_array

    def combine_key_road(self, road_points, key_points, road_angles, key_angles, road_angle_valids, key_angle_valids):
        max_angle_num = max(key_angle_valids.shape[-1], road_angle_valids.shape[-1])

        # padding the angles and angle_valids
        road_pad_size = max_angle_num - road_angles.shape[-1]
        key_pad_size = max_angle_num - key_angles.shape[-1]

        road_angles = torch.nn.functional.pad(road_angles, (0, road_pad_size))
        road_angle_valids = torch.nn.functional.pad(road_angle_valids, (0, road_pad_size))

        key_angles = torch.nn.functional.pad(key_angles, (0, key_pad_size))
        key_angle_valids = torch.nn.functional.pad(key_angle_valids, (0, key_pad_size))

        angles = torch.cat((key_angles, road_angles), dim=0)
        angle_valids = torch.cat((key_angle_valids, road_angle_valids), dim=0)
        node_points = torch.cat((key_points, road_points), dim=0)

        key_node_mask = torch.zeros(len(node_points), dtype=torch.bool, device=node_points.device)
        key_node_mask[:len(key_points)] = True

        return node_points, angles, angle_valids, key_node_mask

    def get_node_nbr(self, node_points):
        device = node_points.device
        num_nodes = node_points.shape[0]

        k = max(0, min(self.graph_infer.max_num_nbrs, num_nodes - 1))
        if num_nodes <= 1:
            return torch.zeros((num_nodes, k), dtype=torch.long, device=device)
        node_points = node_points.cpu().numpy()
        kd_tree = cKDTree(node_points)
        _, nbr_inds = kd_tree.query(node_points, k=k + 1)

        nbr_inds = nbr_inds[:, 1:]
        nbr_inds = torch.from_numpy(nbr_inds).to(device)
        return nbr_inds

    def dual_connection(self, node_points, angles, angle_valids, nbr_inds, connect_valid_mask):
        device, num_nbrs, max_num_angles = node_points.device, nbr_inds.shape[1], angles.shape[1]
        num_centers = connect_valid_mask.sum().item()
        
        center_points, center_angles, center_angle_valids, nbr_inds = \
            node_points[connect_valid_mask], angles[connect_valid_mask], angle_valids[connect_valid_mask], nbr_inds[connect_valid_mask]

        assign_mask = torch.zeros_like(nbr_inds, dtype=torch.bool)
        if num_centers <= 1:
            return assign_mask

        nbr_points = node_points[nbr_inds].view(-1, num_nbrs, 2)
        dist_vec, center_angle_vec = nbr_points - center_points[:, None, :], torch.stack((center_angles.cos(), center_angles.sin()), dim=-1)

        dot_mat = torch.einsum('cnk, cak -> cna', dist_vec, center_angle_vec)  # (num_center, num_nbrs, num_angles)
        dist_mat = torch.norm(dist_vec, dim=-1, keepdim=True).expand(-1, -1, max_num_angles)  # (num_center, num_nbrs, num_angles)
        cos_value = dot_mat / (dist_mat + 1e-6)
        in_range_mask = (cos_value >= self.graph_infer.cos_thres) & (dist_mat < self.graph_infer.r_thres)

        cost_mat_full = dist_mat.clone() # directly use distance value as the cost value 
        cost_mat_full[~in_range_mask] = 1e6
        for c in range(num_centers):
            node_ind, angle_ind = map(lambda x: torch.from_numpy(x).to(device), linear_sum_assignment(cost_mat_full[c, :num_nbrs, center_angle_valids[c]].cpu().numpy()))
            valid_assign = in_range_mask[c, node_ind, angle_ind]
            assign_mask[c, node_ind[valid_assign]] = True
        return assign_mask

    def remove_angles(self, node_points, angles, angle_valids, nbr_ind, assign_mask):
        linked_center_inds, assign_col_ind = torch.nonzero(assign_mask, as_tuple=True)
        linked_nbr_inds = nbr_ind[linked_center_inds, assign_col_ind]

        linked_center_points = node_points[linked_center_inds]
        linked_nbr_points = node_points[linked_nbr_inds]

        linked_vectors = linked_center_points - linked_nbr_points
        linked_angles = angles[linked_nbr_inds]

        linked_angles_vectors = torch.stack((linked_angles.cos(), linked_angles.sin()), dim=-1)
        cos_value = torch.cosine_similarity(linked_vectors.unsqueeze(1), linked_angles_vectors, dim=-1)
        remove_mask_tmp = cos_value > self.graph_infer.remove_cos_thres

        num_angles = angle_valids.shape[1]
        device = angle_valids.device
        
        remove_mask = torch.zeros_like(angle_valids, dtype=torch.bool)
        row_indices = linked_nbr_inds.unsqueeze(1).expand(-1, num_angles)
        col_indices = torch.arange(num_angles, device=device).unsqueeze(0).expand(row_indices.shape[0], -1)
        remove_mask.index_put_((row_indices, col_indices), remove_mask_tmp, accumulate=True)

        angle_valids &= (~remove_mask)
        valid_sort = torch.argsort(angle_valids, dim=-1, descending=True)
        angles = torch.gather(angles, dim=-1, index=valid_sort)
        angle_valids = torch.gather(angle_valids, dim=-1, index=valid_sort)
        return angles, angle_valids














