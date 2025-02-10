# models/pointnet2_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """计算两组点之间的欧氏距离"""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, nsample]
    Return:
        new_points:, indexed points data, [B, S, nsample, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    
    # 添加安全检查
    max_idx = points.shape[1] - 1
    idx = torch.clamp(idx, 0, max_idx)  # 确保索引不会越界
    idx = idx.clamp(0, points.shape[1] - 1)
    new_points = points[batch_indices, idx, :]

    return new_points


def sample_and_group(npoint, radius, nsample, xyz, points):
    """采样和分组操作"""
    B, N, C = xyz.shape
    S = npoint
    
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points

def farthest_point_sample(xyz, npoint):
    """最远点采样"""
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

# def farthest_point_sample(xyz, npoint):
#     """
#     使用AVS-Net替代FPS
#     参数:
#         xyz: 输入点云 (B, N, 3)
#         npoint: 目标点数
#     返回:
#         indices: 采样点的索引 (B, npoint)
#     """
#     avs_net = AVSNet().to(xyz.device)
#     indices = avs_net(xyz, npoint)
#
#     return indices

def query_ball_point(radius, nsample, xyz, new_xyz):
    """球查询"""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    
    return group_idx

class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
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
    
    def forward(self, xyz, points):
        """
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, C, N]
        """
        xyz = xyz.contiguous()
        if points is not None:
            points = points.contiguous()
            points = points.transpose(1, 2)  # 将 [B, C, N] 转换为 [B, N, C]
        
        # Sample and group
        new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        
        # new_points shape: [B, npoint, nsample, C]
        B, npoint, nsample, C = new_points.shape
        new_points = new_points.permute(0, 3, 1, 2).contiguous()  # [B, C, npoint, nsample]
        
        # Point feature embedding
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        # Max pooling
        new_points = torch.max(new_points, -1)[0]  # [B, C, npoint]
        
        return new_xyz, new_points



class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz1: input points position data, [B, N, C]
        xyz2: sampled input points position data, [B, S, C]
        points1: input points data, [B, D, N]
        points2: input points data, [B, D, S]
        """
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        
        if S == 1:
            interpolated_points = points2.repeat(1, 1, N)
        else:
            # 计算 xyz1 和 xyz2 之间的欧几里得距离
            dists = square_distance(xyz1, xyz2)
            # 获取最近的 3 个点
            dists, idx = dists.sort(dim=-1) # 排序，得到距离和索引
            dists, idx = dists[:, :, :3], idx[:, :, :3] # 取最近的 3 个点

            # 计算权重
            dist_recip = 1.0 / (dists + 1e-8)  # 距离的倒数
            norm = torch.sum(dist_recip, dim=2, keepdim=True)  # 归一化因子
            weight = dist_recip / norm  # 归一化权重，形状为 [B, N, 3]

            # 插值特征
            interpolated_points = torch.sum(index_points(points2.transpose(1, 2), idx) * weight.view(B, N, 3, 1), dim=2)

        # Step 2: 特征融合
        if points1 is not None:
            points1 = points1.transpose(1, 2) # 转置为 [B, N, D]
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        # Step 3: 特征更新
        new_points = new_points.transpose(1, 2) # 转置为 [B, D, N]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        return new_points


class EnhancedFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        # 添加注意力机制
        self.attention = nn.Sequential(
            nn.Conv1d(in_channel, in_channel // 4, 1),
            nn.BatchNorm1d(in_channel // 4),
            nn.ReLU(),
            nn.Conv1d(in_channel // 4, in_channel, 1),
            nn.Sigmoid()
        )

        # 添加残差连接
        self.skip_connection = (in_channel == mlp[-1])

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

        # 添加边界感知模块
        self.boundary_aware = nn.Sequential(
            nn.Conv1d(3, 16, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, mlp[-1], 1)
        )

    def forward(self, xyz1, xyz2, points1, points2):
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, 1, N)
        else:
            dists = square_distance(xyz1, xyz2)
            # 增加k近邻数量以提高鲁棒性
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :4], idx[:, :, :4]

            # 改进的距离权重计算
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_points = torch.sum(
                index_points(points2.transpose(1, 2), idx) *
                weight.view(B, N, 4, 1),
                dim=2
            )

        # 特征融合
        if points1 is not None:
            points1 = points1.transpose(1, 2)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.transpose(1, 2)

        # 应用注意力机制
        attention_weights = self.attention(new_points)
        new_points = new_points * attention_weights

        # 边界感知
        edge_features = self.boundary_aware(xyz1.transpose(1, 2))

        # MLP处理
        identity = new_points
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # 残差连接
        if self.skip_connection:
            new_points = new_points + identity

        # 融合边界特征
        new_points = new_points + edge_features

        return new_points


class EnhancedPointTransformerDecoder(nn.Module):
    def __init__(self, in_channel, mlp):
        """
        参数与原FP保持一致:
        in_channel: 输入通道数（包含points1和points2拼接后的通道数）
        mlp: 输出通道数的列表，如[1024, 512]
        """
        super().__init__()
        self.mlp = mlp

        # Q,K,V转换（保持输入维度）
        self.q_conv = nn.Conv1d(in_channel, in_channel, 1)
        self.k_conv = nn.Conv1d(in_channel, in_channel, 1)
        self.v_conv = nn.Conv1d(in_channel, in_channel, 1)

        # MLP层，与原FP结构一致
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

        # 注意力机制（用于特征增强）
        self.attention = nn.Sequential(
            nn.Conv1d(in_channel, in_channel // 4, 1),
            nn.BatchNorm1d(in_channel // 4),
            nn.ReLU(),
            nn.Conv1d(in_channel // 4, in_channel, 1),
            nn.Sigmoid()
        )

        # 边界感知模块
        self.boundary_aware = nn.Sequential(
            nn.Conv1d(3, 16, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, mlp[-1], 1)
        )

        # 残差连接
        self.skip_connection = (in_channel == mlp[-1])

    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz1: (B, N, 3)
        xyz2: (B, S, 3)
        points1: (B, C1, N)
        points2: (B, C2, S)
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, 1, N)
        else:
            # 计算距离和索引
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :4], idx[:, :, :4]  # 取最近的4个点

            # 计算插值权重
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            # Transformer-style特征聚合
            grouped_points = index_points(points2.transpose(1, 2), idx)  # (B, N, k, C)
            grouped_points = grouped_points.permute(0, 3, 1, 2)  # (B, C, N, k)

            # 计算注意力
            q = self.q_conv(points2)  # (B, C, S)
            k = self.k_conv(points2)  # (B, C, S)
            v = self.v_conv(points2)  # (B, C, S)

            grouped_k = index_points(k.transpose(1, 2), idx).permute(0, 3, 1, 2)  # (B, C, N, k)
            grouped_v = index_points(v.transpose(1, 2), idx).permute(0, 3, 1, 2)  # (B, C, N, k)

            # 结合距离权重和注意力权重
            attn = torch.einsum('bcn,bcnk->bnk', q, grouped_k)
            attn = F.softmax(attn / math.sqrt(k.size(1)), dim=-1)  # (B, N, k)
            attn = attn * weight  # 结合距离权重

            # 特征聚合
            interpolated_points = torch.einsum('bnk,bcnk->bcn', attn, grouped_v)

        # 特征拼接（与原FP一致）
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points

        # 通道注意力
        attention_weights = self.attention(new_points)
        new_points = new_points * attention_weights

        # MLP处理
        identity = new_points
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # 边界特征
        edge_features = self.boundary_aware(xyz1.transpose(1, 2))

        # 残差连接
        if self.skip_connection:
            new_points = new_points + identity

        # 融合边界特征
        new_points = new_points + edge_features

        return new_points

# 在 pointnet2_utils.py 中添加
class MultiScaleSetAbstraction(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list

        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        for i in range(len(radius_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            # 考虑到grouped_xyz_norm会添加3个通道
            last_channel = in_channel

            for out_channel in mlp:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel

            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)


    def forward(self, xyz, points):
        """
        Args:
            xyz: (B, N, 3) 坐标
            points: (B, C, N) 特征
        """
        B, N, _ = xyz.shape
        S = self.npoint

        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]

        multi_scale_features = []

        for i, (radius, nsample) in enumerate(zip(self.radius_list, self.nsample_list)):
            idx = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, 3]
            grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, 3)

            if points is not None:
                grouped_points = index_points(points.transpose(1, 2), idx)  # [B, npoint, nsample, C]
                grouped_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz_norm

            grouped_points = grouped_points.permute(0, 3, 1, 2)  # [B, C+3, npoint, nsample]

            for j, (conv, bn) in enumerate(zip(self.conv_blocks[i], self.bn_blocks[i])):
                grouped_points = F.relu(bn(conv(grouped_points)))

            grouped_points = torch.max(grouped_points, -1)[0]  # [B, C, npoint]
            multi_scale_features.append(grouped_points)

        new_points = torch.cat(multi_scale_features, dim=1)
        return new_xyz, new_points


class AVSNet(nn.Module):
    def __init__(self, V0=0.02, Kp=0.5, Ki=0.1, max_iter=10):
        super(AVSNet, self).__init__()
        self.V0 = V0
        self.Kp = Kp
        self.Ki = Ki
        self.max_iter = max_iter
        # 将V0转换为tensor
        self.register_buffer('V0_tensor', torch.tensor(V0, dtype=torch.float))

    def voxel_downsample(self, xyz, voxel_size):
        """体素下采样"""
        B, N, C = xyz.shape
        device = xyz.device

        # 将点云转换为体素坐标
        voxel_coords = torch.floor(xyz / voxel_size).long()

        results = []
        for b in range(B):
            # 为每个体素创建唯一键
            coords = voxel_coords[b]
            unique_keys = coords[:, 0] * 1000000 + coords[:, 1] * 1000 + coords[:, 2]

            # 找到唯一体素
            unique_keys, inverse_indices = torch.unique(unique_keys, return_inverse=True)

            # 计算每个体素的质心
            voxel_points = xyz[b]
            centroids = torch.zeros((len(unique_keys), C), device=device)
            count = torch.zeros(len(unique_keys), device=device)

            # 使用scatter_add_累加体素内的点
            centroids.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, C), voxel_points)
            count.scatter_add_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float))

            # 计算平均值
            centroids = centroids / count.unsqueeze(1).clamp(min=1)
            results.append(centroids)

        # 找到最大长度以便填充
        max_length = max([len(r) for r in results])

        # 填充结果到相同长度
        padded_results = []
        for r in results:
            if len(r) < max_length:
                padding = torch.zeros((max_length - len(r), C), device=device)
                r = torch.cat([r, padding], dim=0)
            padded_results.append(r)

        return torch.stack(padded_results)

    def adapt_voxel_size(self, xyz, npoint):
        """自适应调整体素大小"""
        B, N, _ = xyz.shape
        device = xyz.device
        target_ratio = N / npoint

        # 使用tensor进行计算
        scale = torch.zeros(1, device=device)
        sum_err = torch.zeros(1, device=device)

        for _ in range(self.max_iter):
            # 计算当前体素大小
            voxel_size = self.V0_tensor * torch.exp(scale)

            # 进行体素下采样
            sampled_points = self.voxel_downsample(xyz, voxel_size)
            current_ratio = N / sampled_points.shape[1]

            # 计算误差
            err = target_ratio - current_ratio
            sum_err += err

            # PI控制器调整
            diff = self.Kp * err + self.Ki * sum_err
            scale += 0.01 * (torch.sigmoid(diff) - 0.5)

            # 如果达到目标点数，提前退出
            if abs(sampled_points.shape[1] - npoint) <= npoint * 0.1:
                break

        return voxel_size

    def forward(self, xyz, npoint):
        """
        参数:
            xyz: 输入点云 (B, N, 3)
            npoint: 目标点数
        返回:
            indices: 采样点的索引 (B, npoint)
        """
        B, N, C = xyz.shape
        device = xyz.device

        # 1. 自适应确定体素大小
        voxel_size = self.adapt_voxel_size(xyz, npoint)

        # 2. 执行体素下采样
        sampled_points = self.voxel_downsample(xyz, voxel_size)

        # 3. 为每个采样点找到最近的原始点作为索引
        indices = []
        for b in range(B):
            # 计算距离矩阵
            dist = torch.sum((xyz[b].unsqueeze(1) - sampled_points[b].unsqueeze(0)) ** 2, dim=2)
            # 获取最近点的索引
            batch_indices = torch.argmin(dist, dim=0)

            # 如果采样点数过多，随机选择npoint个点
            if len(batch_indices) > npoint:
                perm = torch.randperm(len(batch_indices), device=device)[:npoint]
                batch_indices = batch_indices[perm]
            # 如果采样点数不足，重复最后一个点
            elif len(batch_indices) < npoint:
                padding = batch_indices[-1].repeat(npoint - len(batch_indices))
                batch_indices = torch.cat([batch_indices, padding])

            indices.append(batch_indices)

        indices = torch.stack(indices)
        return indices


class PointTransformerFusion(nn.Module):
    def __init__(self, feature_dims=[64, 128, 256, 512], output_dim=960, num_heads=8, num_layers=2, dropout=0.1):
        super(PointTransformerFusion, self).__init__()

        # 统一特征维度的转换层
        self.unified_dim = 128  # 统一的特征维度
        self.feature_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, self.unified_dim, 1),
                nn.BatchNorm1d(self.unified_dim),
                nn.ReLU()
            ) for dim in feature_dims
        ])

        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, self.unified_dim, 1))

        # Multi-head Self-Attention
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.unified_dim,
                nhead=num_heads,
                dim_feedforward=self.unified_dim * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # 最终的特征融合层 - 现在输出通道数为output_dim
        self.final_fusion = nn.Sequential(
            nn.Conv1d(self.unified_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, l0_features, l1_features, l2_features, l3_features):
        batch_size = l0_features.shape[0]
        target_points = l0_features.shape[2]  # 使用l0的点数作为目标

        # 转换特征维度并调整点数
        f0 = self.feature_conv[0](l0_features)  # [B, C, N]

        # 对其他特征进行上采样到相同的点数
        f1 = self.feature_conv[1](l1_features)
        f1 = torch.nn.functional.interpolate(f1, size=target_points, mode='linear', align_corners=False)

        f2 = self.feature_conv[2](l2_features)
        f2 = torch.nn.functional.interpolate(f2, size=target_points, mode='linear', align_corners=False)

        f3 = self.feature_conv[3](l3_features)
        f3 = torch.nn.functional.interpolate(f3, size=target_points, mode='linear', align_corners=False)

        # 准备特征序列
        features = torch.stack([
            f0.transpose(1, 2),  # [B, N, C]
            f1.transpose(1, 2),
            f2.transpose(1, 2),
            f3.transpose(1, 2)
        ], dim=2)  # [B, N, 4, C]

        # 重塑为transformer输入格式
        features = features.reshape(batch_size * target_points, 4, self.unified_dim)  # [B*N, 4, C]

        # 添加位置编码
        features = features + self.pos_embedding.transpose(1, 2)

        # Transformer处理
        transformed_features = self.transformer_encoder(features)  # [B*N, 4, C]

        # 重塑并聚合特征
        transformed_features = transformed_features.reshape(batch_size, target_points, 4, self.unified_dim)
        transformed_features = transformed_features.mean(dim=2)  # [B, N, C]
        transformed_features = transformed_features.transpose(1, 2)  # [B, C, N]

        # 最终处理
        output = self.final_fusion(transformed_features)  # [B, output_dim, N]

        return output





