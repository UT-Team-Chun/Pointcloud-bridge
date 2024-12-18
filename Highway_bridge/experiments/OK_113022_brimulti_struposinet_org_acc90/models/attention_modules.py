# models/attention_modules.py
import torch
import torch.nn as nn


# class PositionalEncoding(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.channels = channels
#         self.pos_enc = nn.Sequential(
#             nn.Conv1d(3, channels, 1),
#             nn.BatchNorm1d(channels),
#             nn.ReLU()
#         )
#
#     def forward(self, xyz):
#         # xyz: [B, N, 3]
#         pos_enc = self.pos_enc(xyz.transpose(2, 1))  # [B, C, N]
#         return pos_enc

class PositionalEncoding(nn.Module):
    def __init__(self, channels=64, freq_bands=16):
        super().__init__()
        self.channels = channels
        self.freq_bands = freq_bands

        # Generate sine waves of different frequencies
        freqs = 2.0 ** torch.linspace(0., freq_bands - 1, freq_bands)
        self.register_buffer('freqs', freqs)  # [freq_bands]

        # Initialize the linear layer in __init__
        self.linear_proj = nn.Linear(6 * freq_bands, channels)

    def forward(self, xyz):
        """
        xyz: (B, N, 3) Input point cloud coordinates
        return: (B, N, channels) Position encoding
        """
        # 1. Expand coordinates
        B, N, _ = xyz.shape

        # 2. Calculate sine and cosine encodings at different frequencies
        pos_enc = []
        for freq in self.freqs:
            for func in [torch.sin, torch.cos]:
                pos_enc.append(func(xyz * freq))

        # 3. Concatenate all encodings
        pos_enc = torch.cat(pos_enc, dim=-1)  # [B, N, 6*freq_bands]

        # 4. Project through MLP to specified dimension
        pos_enc = self.linear_proj(pos_enc)  # [B, N, channels]

        # 5. Transpose to match the expected shape
        pos_enc = pos_enc.transpose(1, 2)  # [B, channels, N]

        return pos_enc

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


class EnhancedPositionalEncoding(nn.Module):
    def __init__(self, channels=32, freq_bands=8, k_neighbors=16, max_radius=10.0):
        super().__init__()
        self.channels = channels
        self.k = k_neighbors
        self.freq_bands = freq_bands
        self.max_radius = max_radius

        # 生成不同的频率（用于相对位置编码）
        freqs = 2.0 ** torch.linspace(0., freq_bands - 1, freq_bands)
        self.register_buffer('freqs', freqs)

        # 相对位置编码的MLP
        self.relative_mlp = nn.Sequential(
            nn.Linear(6 * freq_bands + 4, channels // 2),
            nn.LayerNorm(channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels // 2)
        )

        # 结构感知编码的MLP
        self.structure_mlp = nn.Sequential(
            nn.Linear(22, channels // 2),  # 22-dim: 9-dim + 3-dim + 4-dim + 3-dim + 3-dim
            nn.LayerNorm(channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels // 2)
        )

        # 残差连接的投影层
        self.residual_proj = nn.Linear(3, channels)

        # 最终的融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.ReLU()
        )

    def get_relative_encoding(self, xyz, neighbors, center):
        """计算相对位置编码"""
        B, N, k, _ = neighbors.shape

        # 计算相对位置和距离
        rel_pos = neighbors - center  # (B, N, k, 3)
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # (B, N, k, 1)
        diff_normalized = rel_pos / (dist + 1e-8)  # 归一化相对位置

        # 计算频率编码
        pos_enc = []
        for freq in self.freqs:
            for func in [torch.sin, torch.cos]:
                pos_enc.append(func(rel_pos * freq))
        pos_enc = torch.cat(pos_enc, dim=-1)  # (B, N, k, 6*freq_bands)

        # 组合特征
        rel_features = torch.cat([pos_enc, dist, diff_normalized], dim=-1)

        # 通过MLP处理
        encoded = self.relative_mlp(rel_features)  # (B, N, k, channels//2)
        return encoded.mean(dim=2)  # (B, N, channels//2)

    def get_structure_encoding(self, rel_pos):
        """计算增强的结构感知编码
        rel_pos: (B, N, k, 3) 相对位置向量
        """
        B, N, k, _ = rel_pos.shape

        # 1. 局部协方差矩阵特征
        # 计算局部点云的协方差矩阵
        rel_pos_2d = rel_pos.view(B * N, k, 3)
        cov_matrix = torch.bmm(rel_pos_2d.transpose(1, 2), rel_pos_2d) / (k - 1)  # (B*N, 3, 3)
        cov_matrix = cov_matrix.view(B, N, 9)  # 展平协方差矩阵

        # 2. 主方向特征（PCA）
        try:
            # 计算特征值和特征向量
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix.view(B * N, 3, 3))
            eigenvalues = eigenvalues.view(B, N, 3)  # 特征值表示主方向的重要性

            # 计算局部表面的各向异性特征
            anisotropy = (eigenvalues[..., 0] - eigenvalues[..., 2]) / (eigenvalues[..., 0] + 1e-8)
            planarity = (eigenvalues[..., 1] - eigenvalues[..., 2]) / (eigenvalues[..., 0] + 1e-8)
            sphericity = eigenvalues[..., 2] / (eigenvalues[..., 0] + 1e-8)

            # 组合PCA特征
            pca_features = torch.stack([anisotropy, planarity, sphericity], dim=-1)  # (B, N, 3)
        except:
            # 如果PCA计算失败，使用零向量
            pca_features = torch.zeros((B, N, 3), device=rel_pos.device)

        # 3. 局部几何特征
        # 计算每个点到局部中心的距离
        center = rel_pos.mean(dim=2, keepdim=True)  # (B, N, 1, 3)
        distances = torch.norm(rel_pos - center, dim=-1)  # (B, N, k)

        # 计算局部半径和密度
        local_radius = distances.max(dim=-1)[0]  # (B, N)
        point_density = k / (local_radius + 1e-8)  # (B, N)

        # 计算局部曲率（使用距离变化）
        sorted_distances, _ = distances.sort(dim=-1)
        curvature = sorted_distances[..., 1:] - sorted_distances[..., :-1]  # 相邻距离差
        mean_curvature = curvature.mean(dim=-1)  # (B, N)

        # 4. 方向一致性特征
        # 计算相邻点之间的方向变化
        normalized_rel_pos = rel_pos / (torch.norm(rel_pos, dim=-1, keepdim=True) + 1e-8)
        direction_similarity = torch.bmm(
            normalized_rel_pos.view(B * N, k, 3),
            normalized_rel_pos.view(B * N, k, 3).transpose(1, 2)
        ).view(B, N, k, k)
        direction_consistency = direction_similarity.mean(dim=(-1, -2))  # (B, N)

        # 5. 组合所有特征
        geometric_features = torch.stack([
            local_radius,
            point_density,
            mean_curvature,
            direction_consistency
        ], dim=-1)  # (B, N, 4)

        # 计算基本统计特征（保留一些统计信息）
        mean = rel_pos.mean(dim=2)  # (B, N, 3)
        std = rel_pos.std(dim=2)  # (B, N, 3)

        # 组合所有特征
        struct_features = torch.cat([
            cov_matrix,  # 9-dim: 协方差矩阵特征
            pca_features,  # 3-dim: PCA特征
            geometric_features,  # 4-dim: 几何特征
            mean,  # 3-dim: 平均位置
            std,  # 3-dim: 标准差
        ], dim=-1)  # (B, N, 22)

        # 通过MLP处理
        return self.structure_mlp(struct_features)  # (B, N, channels//2)

    def forward(self, xyz):
        """
        xyz: (B, N, 3) 输入点云坐标
        return: (B, channels, N) 位置编码
        """
        B, N, _ = xyz.shape
        device = xyz.device

        # 1. 找到k近邻
        dist = torch.cdist(xyz, xyz)  # (B, N, N)
        k = min(self.k, N)
        _, idx = dist.topk(k, dim=-1, largest=False)  # (B, N, k)

        # 获取近邻坐标
        idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
        idx = idx + idx_base
        idx = idx.view(-1)

        neighbors = xyz.view(B * N, -1)[idx].view(B, N, k, -1)  # (B, N, k, 3)
        center = xyz.unsqueeze(2).expand(-1, -1, k, -1)  # (B, N, k, 3)

        # 2. 计算相对位置编码
        rel_encoding = self.get_relative_encoding(xyz, neighbors, center)  # (B, N, channels//2)
        #print(f"rel_encoding shape before fusion: {rel_encoding.shape}")

        # 3. 计算结构感知编码
        rel_pos = neighbors - center
        struct_encoding = self.get_structure_encoding(rel_pos)  # (B, N, channels//2)
        #print(f"struct_encoding shape before fusion: {struct_encoding.shape}")

        # 4. 组合所有特征
        combined_encoding = torch.cat([rel_encoding, struct_encoding], dim=-1)  # (B, N, channels)
        #print(f"combined_encoding shape before fusion: {combined_encoding.shape}")
        #encoded = self.fusion_layer(combined_encoding)  # (B, N, channels)
        final_encoding = combined_encoding

        return final_encoding.transpose(1, 2)  # (B, channels, N)


if __name__ == '__main__':
    # 测试PositionalEncoding
    xyz = torch.randn(2, 4096, 3) #B, N, C
    # 实例化 EnhancedPositionalEncoding 类
    pos_enc = EnhancedPositionalEncoding(channels=32, freq_bands=16, k_neighbors=3)

    # 调用 forward 方法
    output = pos_enc(xyz)

    # 打印输出形状
    print(f"Output shape: {output.shape}")