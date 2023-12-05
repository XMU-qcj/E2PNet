import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
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
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
#     dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
#     dist += torch.sum(src ** 2, -1).view(B, N, 1)
#     dist += torch.sum(dst ** 2, -1).view(B, 1, M)
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

    dist = (1/(torch.sqrt(2 * dist_t + 1e-6) + 1e-6)) + (1/(torch.sqrt(dist_xy + 1e-6) + 1e-6))
    # src_dist = src[:, :, :3]
    # dst_dist = dst[:, :, :3]
    # dist = -2 * torch.matmul(src_dist, dst_dist.permute(0, 2, 1))
    # dist += torch.sum(src_dist ** 2, -1).view(B, N, 1)
    # dist += torch.sum(dst_dist ** 2, -1).view(B, 1, M)
    return dist_t, dist_xy, dist, dist_ori

# def square_distance(src, dst):
#     """
#     Calculate Euclid distance between each two points.
#     src^T * dst = xn * xm + yn * ym + zn * zm；
#     sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
#     sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
#     dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
#          = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
#     Input:
#         src: source points, [B, N, C]
#         dst: target points, [B, M, C]
#     Output:
#         dist: per-point square distance, [B, N, M]
#     """
#     B, M, _ = dst.shape
    
#     src_dist = src[:, :, 2].unsqueeze(2)
#     dst_dist = dst[:, :, 2].unsqueeze(1)
    
#     dist_t = torch.matmul(src_dist, dst_dist)
    
#     src_dist = src[:, :, :2]
#     dst_dist = dst[:, :, :2]
    
#     dist_xy = torch.matmul(src_dist, dst_dist.transpose(2,1))
    
# #     dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
# #     dist += torch.sum(src ** 2, -1).view(B, N, 1)
# #     dist += torch.sum(dst ** 2, -1).view(B, 1, M)
#     return torch.abs(dist_t), torch.abs(dist_xy)


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


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


# def query_ball_point(radius, nsample, xyz, new_xyz, t_list):
#     """
#     Input:
#         radius: local region radius
#         nsample: max sample number in local region
#         xyz: all points, [B, N, 3]
#         new_xyz: query points, [B, S, 3]
#     Return:
#         group_idx: grouped points index, [B, S, nsample]
#     """
#     device = xyz.device
#     B, N, C = xyz.shape
#     _, S, _ = new_xyz.shape
    
    
#     src_dist = xyz[:, :, 2].unsqueeze(2)
#     dst_dist = new_xyz[:,:4096,2].unsqueeze(1)
    
#     dist = torch.abs(torch.matmul(src_dist, dst_dist)) # 2,20000, 5
#     print(4,dist,dist.shape)
#     dist = dist.sort(-1)[1] # 2,20000
#     print(2,dist, dist.shape)
    
#     pixel = torch.arange(0, 256, 4, dtype=torch.long).to(device)
#     print(3,pixel, pixel.shape)
#     print(11111111111111111111)
#     print(dsadsadsd)
    
    
    
    
    
#     group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
#     pixel = torch.arange(0, 256, 4, dtype=torch.long).to(device)
#     print(pixel, pixel.shape)
#     sqrdists = square_distance(t_list, xyz)
#     val = (torch.arange(0, 256, 4)).to(xyz.device)
#     x_val = (val.unsqueeze(-1).repeat(1, 64)).reshape((-1))
#     print(x_val,x_val.shape)
#     print(666, group_idx, group_idx.shape, sqrdists.shape)
#     print(dsads)
    
#     group_idx[sqrdists > radius ** 2] = N
#     group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
#     group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
#     mask = group_idx == N
#     group_idx[mask] = group_first[mask]
#     return group_idx

def query_ball_point(radius, nsample, xyz, new_xyz, t_list, i = 0):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
#     group_idx = torch.arange(N, dtype=torch.long)
#     group_idx = group_idx.cuda()
#     group_idx = group_idx.view(1, 1, N)
#     group_idx = group_idx.repeat([B, S, 1])

    dist_1 = t_list[:, 1] - t_list[:,0]
    dist_2 = t_list[:, 2] - t_list[:,1]
    dist_3 = t_list[:, 3] - t_list[:,2]
    dist_4 = t_list[:, 4] - t_list[:,3]
    dist_5 = t_list[:, 5] - t_list[:,4]
    dist_6 = t_list[:, 6] - t_list[:,5]
    # t_1 = dist_1.unsqueeze(-1)
    # t_2 = dist_2.unsqueeze(-1)
    # t_3 = dist_3.unsqueeze(-1)
    # t_4 = dist_4.unsqueeze(-1)
    # t_5 = dist_5.unsqueeze(-1)
    # t_6 = dist_6.unsqueeze(-1)
    t_1 = (dist_1 + dist_2).unsqueeze(-1)
    t_2 = (dist_2 + dist_3).unsqueeze(-1)
    t_3 = (dist_3 + dist_4).unsqueeze(-1)
    t_4 = (dist_4 + dist_5).unsqueeze(-1)
    t_5 = (dist_5 + dist_6).unsqueeze(-1)
    t = torch.cat([t_1, t_2, t_3, t_4, t_5], -1)
    max_t = t.sort(dim=-1, descending = True)[0]#((torch.min(t, -1)[0]) ** 2).unsqueeze(-1).unsqueeze(-1)
    max_t = (max_t[:, i] ** 2).unsqueeze(-1).unsqueeze(-1)

    # dist_t, dist_xy, dist, dist_ori = square_distance(new_xyz, xyz)
    # _, group_idx = dist_ori.sort(dim=-1)
    # group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # dist[dist_xy > radius] = -N
    # dist[dist_t > max_t] = -N
    # sort_dis, group_idx = dist.sort(dim=-1, descending=True)
    # mask = sort_dis[:,:,:nsample] == -N
    # group_idx = group_idx[:, :, :nsample]
    # group_idx[mask] = group_first[mask]

    # dist_t, dist_xy, dist, dist_ori = square_distance(new_xyz, xyz)
    # _, group_idx = dist_ori.sort(dim=-1)
    # group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # dist[dist_xy > radius] = N
    # dist[dist_t > max_t] = N
    # sort_dis, group_idx = dist_ori.sort(dim=-1)
    # mask = sort_dis[:,:,:nsample] == N
    # group_idx = group_idx[:, :, :nsample]
    # group_idx[mask] = group_first[mask]

    # group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    # dist_t, dist_xy, dist, dist_ori = square_distance(new_xyz, xyz)
    # group_idx[dist_xy > radius] = N -1
    # group_idx[dist_t > max_t] = N -1
    # group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # # _, group_idx_ori = dist_ori.sort(dim=-1)
    # group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # # group_first = group_idx_ori[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # mask = group_idx == N -1
    # group_idx[mask] = group_first[mask]

    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    dist_t, dist_xy, dist, dist_ori = square_distance(new_xyz, xyz)
    group_idx[dist_xy > radius] = N
    group_idx[dist_t > max_t] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    _, group_idx_ori = dist_ori.sort(dim=-1)
    group_first = group_idx_ori[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx
    
    
def sample_and_group(radius, nsample, xyz, points, new_xyz, t_list, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    
#     print(111,xyz.shape, new_xyz.shape) # B, N, 4,  B, M, 4
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    idx = query_ball_point(radius, nsample, xyz, new_xyz, t_list)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]

    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
        
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points, new_xyz, t_list):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
#     print(xyz.shape, points.shape)
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points, mask, t_list):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points, mask, t_list)
        else:
            new_xyz, new_points = sample_and_group(self.radius, self.nsample, xyz, points, mask, t_list)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points, mask, t_list):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B, N, C = xyz.shape
        _, S, _ = mask.shape
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, mask, t_list, (i * 2))
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= mask.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = mask.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
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
        
        xyz1 = xyz1.transpose(2,1)
        points1 = points1.transpose(2,1)

        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            _, _, _, dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points