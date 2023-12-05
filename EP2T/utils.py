import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def square_distance(src, dst, weight):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
        weight: weight between space and time domain
    Output:
        dist_t: per-point temporal distance, [B, N, M]
        dist_xy: per-point spatial distance, [B, N, M]
        dist: per-point weighted square distance, [B, N, M]
        dist_ori: per-point square distance, [B, N, M]        
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape

    src_dist = src[:, :, :2]
    dst_dist = dst[:, :, :2]
    dist_xy = -2 * torch.matmul(src_dist, dst_dist.permute(0, 2, 1))
    dist_xy += torch.sum(src_dist ** 2, -1).view(B, N, 1)
    dist_xy += torch.sum(dst_dist ** 2, -1).view(B, 1, M)

    src_dist = src[:, :, 2].unsqueeze(-1)
    dst_dist = dst[:, :, 2].unsqueeze(-1)
    dist_t = -2 * torch.matmul(src_dist, dst_dist.permute(0, 2, 1))
    dist_t += torch.sum(src_dist ** 2, -1).view(B, N, 1)
    dist_t += torch.sum(dst_dist ** 2, -1).view(B, 1, M)
    
    src_dist = src[:, :, :3]
    dst_dist = dst[:, :, :3]
    dist_ori = -2 * torch.matmul(src_dist, dst_dist.permute(0, 2, 1))
    dist_ori += torch.sum(src_dist ** 2, -1).view(B, N, 1)
    dist_ori += torch.sum(dst_dist ** 2, -1).view(B, 1, M)
    dist = dist_xy * weight[0] + dist_t * weight[1] 

    return dist_t, dist_xy, dist, dist_ori

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).cuda().view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def query_ball_point(radius, nsample, xyz, new_xyz, t_list, weight):
    """
    Input:
        radius: local region radius  #radius= (12/256.) ** 2
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
        t_list: all sampled time, [B, n_time]
        weight: weights between time and space, [[x1,y1], [x2,y2]]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    
    distances = t_list[:, 1:] - t_list[:, :-1]
    max_t = distances.sort(dim=-1, descending = True)[0]
    max_t = (max_t[:, 0] ** 2).unsqueeze(-1).unsqueeze(-1)
    
    dist_t, dist_xy, dist, dist_ori = square_distance(new_xyz, xyz, weight)
    _, group_idx = dist_ori.sort(dim=-1)
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    dist[dist_xy > radius] = N
    dist[dist_t > max_t] = N
    sort_dis, group_idx = dist.sort(dim=-1)
    mask = sort_dis[:,:,:nsample] == N 
    group_idx = group_idx[:, :, :nsample]
    group_idx[mask] = group_first[mask]

    return group_idx
    
    
def sample_and_group(radius, nsample, xyz, points, new_xyz, t_list, weight):
    """
    Input:
        radius: max distance between center and its corresponding neighbour points
        nsample: max number of neighbour points
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
        new_xyz: input center points position, [B, M, 3]
        t_list: all sampled time, [B, n_time]
        weight: weights between time and space, [[x1,y1], [x2,y2]]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    
    idx = query_ball_point(radius, nsample, xyz, new_xyz, t_list, weight)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]

    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, radius, nsample, in_channel, mlp, weight):
        super(PointNetSetAbstraction, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.module = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.module.append(nn.Conv2d(last_channel, out_channel, 1))
            self.module.append(nn.BatchNorm2d(out_channel))
            self.module.append(nn.ReLU())
            last_channel = out_channel
        self.weight = weight

    def forward(self, xyz, points, center, t_list):
        res_points = []
        res_xyz = []
        for i in range(len(self.weight)):
            new_xyz, new_points = sample_and_group(self.radius, self.nsample, xyz, points, center, t_list, self.weight[i]) 
            new_points = new_points.permute(0, 3, 2, 1)
            for layer in self.module:
                new_points = layer(new_points)
            new_points = torch.max(new_points, 2)[0] 
            new_xyz = new_xyz.permute(0, 2, 1)
            res_xyz.append(new_xyz)
            res_points.append(new_points)
        res_xyz = torch.stack(res_xyz, 1)
        res_points = torch.stack(res_points, 1)
        return res_xyz, res_points
    
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.module = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.module.append(nn.Conv1d(last_channel, out_channel, 1))
            self.module.append(nn.BatchNorm1d(out_channel))
            self.module.append(nn.ReLU())
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            _, _, _, dists = square_distance(xyz1, xyz2, [0.5, 0.5])
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :4], idx[:, :, :4]  # [B, N, 3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 4, 1), dim=2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for layer in self.module:
            new_points = layer(new_points)
        return new_points