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
    centroid = np.mean(pc, axis=0) # xyz
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1))) # max distance
    pc = pc / m
    return pc


def square_distance(src, dst):
    """
    Calculate Euclidean distance between each two points.

    src.T * dst = xn * xm + yn * ym + zn * zm
    sum(src ^ 2, dim=-1) = xn * xn + yn * yn + zn * zn
    sum(dst ^ 2, dim=-1) = xm * xm + ym * ym + zm * zm
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2, dim=-1) + sum(dst**2, dim=-1) - 2*src.T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Select idx from points
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] (or [B, M, S])
    Return:
        new_points: indexed points data, [B, S, C] (or [B, M, S, C])
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape) # [B, S] (or [B, M, S])
    view_shape[1: ] = [1] * (len(view_shape) - 1) # [B, 1] (or [B, 1, 1])
    repeat_shape = list(idx.shape) # [B, S] (or [B, M, S])
    repeat_shape[0] = 1 # [1, S] (or [1, M, S])
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape) # [B, S] (or [B, M, S])
    new_points = points[batch_indices, idx, : ]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Select npoint points from xyz
    Input:
        xyz: pointcloud data, [B, N, d], usually d = 3
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, d = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device) # [B, npoint]
    distance = torch.ones(B, N).to(device) * 1e10 # [B, N]
    farthest = torch.randint(low=0, high=N, size=(B, ), dtype=torch.long).to(device) # batch里每个样本随机初始化一个最远点的索引
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[: , i] = farthest # 第一个采样点选随机初始化的索引
        centroid = xyz[batch_indices, farthest, : ].view(B, 1, 3) # 得到当前采样点的坐标[B, 1, 3]
        dist = torch.sum((xyz - centroid) ** 2, -1) # 计算当前采样点与其他点的距离
        mask = dist < distance # 选择距离最近的来更新距离（更新维护这个表）
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1] # 重新计算得到最远点索引（在更新的表中选择距离最大的那个点）
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points coordinate data, [B, N, d], usually d = 3
        new_xyz: query points coordinate data (centroids obtained by FPS), [B, S, d]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, d = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1]) # [B, S, N]
    sqrdists = square_distance(new_xyz, xyz) # 得到[B, S, N]（就是S个点中每一个和N中每一个的欧氏距离）
    group_idx[sqrdists > radius ** 2] = N # 找到距离大于给定半径的设置成一个N值（1024）索引
    group_idx = group_idx.sort(dim=-1)[0][: , : , : nsample] # 做升序排序，后面的都是大的值（1024）, [B, S, nsample]
    group_first = group_idx[: , : , 0].view(B, S, 1).repeat([1, 1, nsample]) # 如果半径内的点没那么多，就直接用第一个点来代替
    # mask = (group_idx == N) !!!
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


# select S (npoint) points from N points using FPS
def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint: number of samples
        radius: local region radius
        nsample: max sample number in local region
        xyz: input points coordinate data, [B, N, d], usually d = 3
        points: input points data (coordinates and feature), [B, N, D], D = d + C
    Return:
        new_xyz: sampled points coordinate data (centroids obtained by FPS), [B, S, d]
        new_points: sampled points data (coordinates and feature), [B, S, nsample, 2 * d + C = d + D]
    """
    B, N, d = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, S) # index, [B, S]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx) # locate (centroids obtained by FPS)
    torch.cuda.empty_cache()
    # idx: indices for points in each spherical region [B, S, nsample]
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, S, nsample, d]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, d)
    torch.cuda.empty_cache()

    # if there are new dimensions (feature) for each point, segment them with coordinates, otherwise just return the coordinates
    if points is not None:
        grouped_points = index_points(points, idx) # [B, S, nsample, D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, S, nsample, d + D]
    else:
        new_points = grouped_xyz_norm

    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


# Warning: input shape is [B, N, D], not [B, D, N]!
# all points in ONE group, npoint = S = 1, nsample = N!
def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points coordinate data, [B, N, d], usually d = 3
        points: input points data (coordinates and feature), [B, N, D], D = d + C
    Return:
        new_xyz: sampled points coordinate data (centroids obtained by FPS), [B, 1, d]
        new_points: sampled points data, [B, 1, N, d + D]
    """
    device = xyz.device
    B, N, d = xyz.shape
    new_xyz = torch.zeros(B, 1, d).to(device) # use the coordinate origin as centroids 
    grouped_xyz = xyz.view(B, 1, N, d)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


# SetAbstraction = (sampling & grouping) + pointnet
# 'mlp' is a list ([32, 32, 64])
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
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

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points coordinate data, [B, d, N], usually d = 3
            points: input points data (coordinates and feature), [B, D, N], D = d + C
        Return:
            new_xyz: sampled points coordinate data (centroids obtained by FPS), [B, d, npoint]
            new_points: sample points data (coordinates and feature), [B, last channel in mlp, npoint]
        """
        xyz = xyz.permute(0, 2, 1) # output.shape: [B, N, d]
        if points is not None:
            points = points.permute(0, 2, 1) # output.shape: [B, N, D]
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
            # new_xyz: sampled points coordinate data (centroids obtained by FPS), [B, npoint, d]
            # new_points: sampled points data, [B, npoint, nsample, (2 * d + C = d + D) --> (in_channel)]
        new_points = new_points.permute(0, 3, 2, 1) # [B, (d + D) --> (in_channel), nsample, npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        # new_points.shape: [B, last channel in mlp, nsample, npoint]
        new_points = torch.max(new_points, 2)[0] # output.shape: [B, last channel in mlp, npoint]
        new_xyz = new_xyz.permute(0, 2, 1) # output.shape: [B, d, npoint]
        return new_xyz, new_points # [B, d, npoint], [B, last channel in mlp, npoint]


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points coordinate data, [B, d, N], usually d = 3
            points: input points data (coordinates and feature), [B, D, N], D = d + C
        Return:
            new_xyz: sampled points coordinate data (centroids obtained by FPS), [B, d, npoint]
            new_points_concat: sample points feature data, [B, sum (last channel in mlp[i]), npoint]
        """
        xyz = xyz.permute(0, 2, 1) # output.shape: [B, N, d]
        if points is not None:
            points = points.permute(0, 2, 1) # output.shape: [B, N, D]
        B, N, d = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S)) # select S points from xyz using FPS, [B, S, d]

        new_points_list = [] # features in different radii
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i] # like the 'nsample' before: max sample number in local region
            group_idx = query_ball_point(radius, K, xyz, new_xyz) # indices, [B, S, K]
            grouped_xyz = index_points(xyz, group_idx) # [B, S, K, d]
            grouped_xyz -= new_xyz.view(B, S, 1, d) # normalize
            if points is not None:
                grouped_points = index_points(points, group_idx) # [B, S, K, D]
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1) # output.shape: [B, S, K, d + D]
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, d + D, K, S]

            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            
            # grouped_points.shape: [B, last channel in mlp[i], K, S]
            new_points = torch.max(grouped_points, 2)[0] # [B, last channel in mlp[i], S]

            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1) # output.shape: [B, d, S]
        new_points_concat = torch.cat(new_points_list, dim=1) # output.shape: [B, sum(last channel in mlp[i]), S]

        return new_xyz, new_points_concat # [B, d, S], [B, sum (last channel in mlp[i]), S]
    

# UPsample!
class PointNetFeaturePropagation(nn.Module):
    # e.g. in_channel = 768, mlp = [256, 256]
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1)) # here is '1d'!
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz2  ---interpolate--->  xyz1
        (xyz2 and points2) are obtained from (xyz1 and points1) by PointNetSetAbstraction
        Input:
            xyz1: input points coordinate data, [B, d, N], usually d = 3, more points, N > S
            xyz2: sampled input points coordinate data (centroids obtained by FPS), [B, d, S], less points, N > S
            points1: input points data, [B, D1, N], D1 + D2 = in_channel
            points2: input points data, [B, D2, S], D1 + D2 = in_channel
        Return:
            new_points: upsampled points data, [B, last channel in mlp, N]
        """
        xyz1 = xyz1.permute(0, 2, 1) # [B, N, d]
        xyz2 = xyz2.permute(0, 2, 1) # [B, S, d]
        points2 = points2.permute(0, 2, 1) # [B, S, D2]

        B, N, d = xyz1.shape
        _, S, _ = xyz2.shape

        # just copy the point if only ONE point is left
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1) # [B, N, D2]
        # do linear interpolation if more than ONE points are left
        else:
            # in the paper: 'in default we use p = 2, k = 3', p is the power of distance and k is the number of top closest points
            dists = square_distance(xyz1, xyz2) # [B, N, S]
            # 'idx' is index in xyz2 (or points2)
            dists, idx = dists.sort(dim=-1) # point-wise distance from xyz1 to xyz2 (near to far)
            dists, idx = dists[: , : , : 3], idx[: , : , : 3]  # [B, N, 3], the top three closest points, in default by the paper

            dist_recip = 1.0 / (dists + 1e-8) # reciprocal of distance ('wi(x)' in the paper), longer distance means smaller weight
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm # [B, N, 3]
            selected_3_points_in_points2 = index_points(points2, idx) # [B, N, 3, D2]
            interpolated_points = torch.sum(selected_3_points_in_points2 * weight.view(B, N, 3, 1), dim=2) # [B, N, D2]

        if points1 is not None:
            points1 = points1.permute(0, 2, 1) # [B, N, D1]
            new_points = torch.cat([points1, interpolated_points], dim=-1) # [B, N, D1 + D2], D1 + D2 = in_channel
        else:
            new_points = interpolated_points # [B, N, D2], D2 = in_channel

        new_points = new_points.permute(0, 2, 1) # [B, D1 + D2, N] (or [B, D2, N] if points1 is None)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        # new_points.shape: [B, last channel in mlp, N]

        return new_points