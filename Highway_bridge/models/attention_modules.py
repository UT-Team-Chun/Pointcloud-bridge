# models/attention_modules.py
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.pos_enc = nn.Sequential(
            nn.Conv1d(3, channels, 1),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )

    def forward(self, xyz):
        # xyz: [B, N, 3]
        pos_enc = self.pos_enc(xyz.transpose(2, 1))  # [B, C, N]
        return pos_enc

# class PositionalEncoding(nn.Module):
#     def __init__(self, channels=64, freq_bands=16):
#         super().__init__()
#         self.channels = channels
#         self.freq_bands = freq_bands
#
#         # Generate sine waves of different frequencies
#         freqs = 2.0 ** torch.linspace(0., freq_bands - 1, freq_bands)
#         self.register_buffer('freqs', freqs)  # [freq_bands]
#
#         # Initialize the linear layer in __init__
#         self.linear_proj = nn.Linear(6 * freq_bands, channels)
#
#     def forward(self, xyz):
#         """
#         xyz: (B, N, 3) Input point cloud coordinates
#         return: (B, N, channels) Position encoding
#         """
#         # 1. Expand coordinates
#         B, N, _ = xyz.shape
#
#         # 2. Calculate sine and cosine encodings at different frequencies
#         pos_enc = []
#         for freq in self.freqs:
#             for func in [torch.sin, torch.cos]:
#                 pos_enc.append(func(xyz * freq))
#
#         # 3. Concatenate all encodings
#         pos_enc = torch.cat(pos_enc, dim=-1)  # [B, N, 6*freq_bands]
#
#         # 4. Project through MLP to specified dimension
#         pos_enc = self.linear_proj(pos_enc)  # [B, N, channels]
#
#         # 5. Transpose to match the expected shape
#         pos_enc = pos_enc.transpose(1, 2)  # [B, channels, N]
#
#         return pos_enc

    def to(self, device):
        """
        Override to method to ensure all components move to the same device
        """
        super().to(device)
        self.linear_proj = self.linear_proj.to(device)
        return self


class BoundaryAwareModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # 边界特征提取
        self.boundary_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 4, 1),
            nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(),
            nn.Conv1d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, xyz):
        """
        x: [B, C, N] 特征
        xyz: [B, N, 3] 坐标
        """
        # 提取边界特征
        boundary_feat = self.boundary_conv(x)

        # 计算注意力权重
        attention_weights = self.attention(x)

        # 应用注意力
        enhanced_feat = x + boundary_feat * attention_weights

        return enhanced_feat


# 在 attention_modules.py 中添加
class EnhancedAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv1d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 4, 1),
            nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(),
            nn.Conv1d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [B, C, N]
        """
        # 通道注意力
        ca = self.channel_attention(x)
        x_ca = x * ca

        # 空间注意力
        sa = self.spatial_attention(x_ca)
        x_sa = x_ca * sa

        return x_sa


class GeometricFeatureExtraction(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels + 3, in_channels, 1),  # 修改输入通道数为 in_channels + 3
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels, 1)
        )

    def forward(self, x, xyz):
        """
        提取几何特征
        x: [B, C, N] 特征
        xyz: [B, N, 3] 坐标
        """
        # 计算法向量
        normals = compute_normals(xyz)  # [B, N, 3]

        # 合并特征
        geometric_features = torch.cat([
            x,
            normals.transpose(1, 2)  # [B, 3, N]
        ], dim=1)  # [B, C+3, N]

        return self.mlp(geometric_features)



def square_distance(src, dst):
    """
    计算两组点之间的成对平方距离。

    参数:
        src: 源点，形状为(B, N, C)
        dst: 目标点，形状为(B, M, C)
    返回:
        dist: 成对距离矩阵，形状为(B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    根据索引从点云中获取对应的点

    参数:
        points: 输入点云，形状为(B, N, C)
        idx: 索引，形状为(B, N, k)
    返回:
        indexed_points: 索引后的点，形状为(B, N, k, C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def compute_normals(xyz):
    """
    计算点云法向量
    xyz: [B, N, 3]
    return: [B, N, 3]
    """
    B, N, _ = xyz.shape
    device = xyz.device

    # 使用KNN找到临近点
    k = 20
    dist = square_distance(xyz, xyz)
    idx = dist.topk(k=k, dim=-1, largest=False)[1]  # [B, N, k]

    # 获取邻域点
    neighbors = index_points(xyz, idx)  # [B, N, k, 3]

    # 计算协方差矩阵
    neighbors = neighbors - xyz.unsqueeze(2)  # 中心化
    covariance = torch.matmul(neighbors.transpose(2, 3), neighbors)  # [B, N, 3, 3]

    # 特征值分解
    eigenvals, eigenvects = torch.linalg.eigh(covariance)

    # 取最小特征值对应的特征向量作为法向量
    normals = eigenvects[..., 0]  # [B, N, 3]

    return normals
