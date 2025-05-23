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

        # 边界特征提取网络
        self.boundary_net = nn.Sequential(
            nn.Conv1d(in_channels * 2, in_channels, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )

        # 空间关系编码
        self.spatial_relation = nn.Sequential(
            nn.Conv1d(4, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1)
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels + 64, in_channels // 2, 1),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(),
            nn.Conv1d(in_channels // 2, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, xyz):
        """
        x: [B, C, N] 特征
        xyz: [B, N, 3] 坐标
        """
        B, C, N = x.shape

        # 计算点对之间的相对位置和距离
        xyz_transpose = xyz.transpose(1, 2)  # [B, 3, N]

        # 计算局部特征
        inner_product = torch.matmul(xyz, xyz.transpose(1, 2))
        xx = torch.sum(xyz ** 2, dim=2, keepdim=True)
        distance = xx + xx.transpose(1, 2) - 2 * inner_product  # [B, N, N]

        # 选择K个最近邻
        K = 16
        knn_idx = torch.topk(distance, k=K, dim=-1, largest=False)[1]  # [B, N, K]

        # 获取相对坐标和距离特征
        batch_idx = torch.arange(B, device=xyz.device).view(-1, 1, 1)
        knn_xyz = xyz[batch_idx, knn_idx, :]  # [B, N, K, 3]
        relative_xyz = knn_xyz - xyz.unsqueeze(2)  # [B, N, K, 3]
        relative_distance = torch.norm(relative_xyz, dim=-1, keepdim=True)  # [B, N, K, 1]

        # 计算平均相对位置和距离，并调整维度
        mean_relative_xyz = torch.mean(relative_xyz, dim=2)  # [B, N, 3]
        mean_relative_xyz = mean_relative_xyz.permute(0, 2, 1)  # [B, 3, N]

        mean_distance = torch.mean(relative_distance, dim=2)  # [B, N, 1]
        mean_distance = mean_distance.permute(0, 2, 1)  # [B, 1, N]

        # 准备空间关系特征
        spatial_features = torch.cat([
            mean_relative_xyz,  # [B, 3, N]
            mean_distance  # [B, 1, N]
        ], dim=1)  # [B, 4, N]

        # 提取空间关系特征
        spatial_feat = self.spatial_relation(spatial_features)  # [B, 64, N]

        # 获取局部特征差异
        knn_features = x.unsqueeze(2).expand(-1, -1, N, -1)  # [B, C, N, N]
        knn_features = torch.gather(knn_features, 3,
                                    knn_idx.unsqueeze(1).expand(-1, C, -1, -1))  # [B, C, N, K]

        # 计算局部特征差异
        local_diff = knn_features - x.unsqueeze(3)  # [B, C, N, K]

        # 合并局部特征
        boundary_feat = torch.cat([
            x,
            torch.max(local_diff, dim=3)[0]  # 最大池化聚合
        ], dim=1)  # [B, 2C, N]

        # 提取边界特征
        boundary_feat = self.boundary_net(boundary_feat)

        # 计算注意力权重
        attention_input = torch.cat([x, spatial_feat], dim=1)
        attention_weights = self.attention(attention_input)

        # 特征增强
        enhanced_feat = x + boundary_feat * attention_weights

        return enhanced_feat


class StructuralAwareModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # 结构特征提取
        self.structure_net = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels, 1)
        )

        # 全局上下文
        self.global_context = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Conv1d(in_channels, in_channels // 4, 1),
            nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(),
            nn.Conv1d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 提取结构特征
        struct_feat = self.structure_net(x)

        # 全局上下文注意力
        context = self.global_context(x)

        return x + struct_feat * context



# 在 attention_modules.py 中添加
class EnhancedAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # 局部特征变换
        self.local_transform = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels, 1)
        )

        # 最大池化通道注意力
        self.max_channel_attention = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Conv1d(in_channels, in_channels // 4, 1),
            nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(),
            nn.Conv1d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

        # 平均池化通道注意力
        self.avg_channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // 4, 1),
            nn.BatchNorm1d(in_channels // 4),
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

        # 可学习权重
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 局部特征变换
        local_feat = self.local_transform(x)

        # 融合最大池化和平均池化的通道注意力
        max_att = self.max_channel_attention(x)
        avg_att = self.avg_channel_attention(x)
        channel_att = max_att + avg_att

        # 应用通道注意力
        x_ca = x * channel_att

        # 空间注意力
        spatial_att = self.spatial_attention(x_ca)
        x_sa = x_ca * spatial_att

        # 组合所有特征
        output = x + self.gamma1 * local_feat + self.gamma2 * x_sa

        return output

class GeometricFeatureExtraction(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels + 16, in_channels, 1),  # 修改输入通道数为 in_channels + 3
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels, 1)
        )

        self.br_pos=BridgeStructureEncoding(channels=16)

    def forward(self, x, xyz):
        """
        提取几何特征
        x: [B, C, N] 特征
        xyz: [B, N, 3] 坐标
        """
        # 计算法向量
        #normals = compute_normals(xyz)  # [B, N, 3]
        pos_enc=self.br_pos(xyz) #[B,C,N]

        # 合并特征
        geometric_features = torch.cat([
            x,
            pos_enc #.transpose(1, 2)  # [B, 3, N]
        ], dim=1)  # [B, C+16, N]

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
    def __init__(self, channels=32, freq_bands=4, k_neighbors=16, max_radius=10.0):
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
            nn.Conv2d(6 * freq_bands + 4, channels // 2, 1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels // 2, 1)
        )

        # 结构感知编码的MLP
        self.structure_mlp = nn.Sequential(
            nn.Conv2d(22, channels // 2, 1),  # 22-dim: 9-dim + 3-dim + 4-dim + 3-dim + 3-dim
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels // 2, 1)
        )

        # 残差连接的投影层
        self.residual_proj = nn.Linear(3, channels)

        # 最终的融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.ReLU(),
            nn.Dropout(0.1)
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
        rel_features = rel_features.permute(0, 3, 1, 2)  # (B, N, k, channels//2) → (B, channels//2, N, k)
        encoded = self.relative_mlp(rel_features)  # [B, C//2, N, K]
        encoded = encoded.permute(0, 2, 3, 1)  # [B, N, K, channels//2]

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

        struct_features = struct_features.permute(0, 2, 1).unsqueeze(2)  # (B, 22, N) -> (B, 22, 1, N)
        struct_features = self.structure_mlp(struct_features)  # (B, 22, 1, N) -> (B, channels//2, 1, N)
        struct_features = struct_features.squeeze(2)  # (B, channels//2, 1, N) -> (B, channels//2, N)
        return struct_features.permute(0, 2, 1)  # 添加这行，将 (B, channels//2, N) 转换为 (B, N, channels//2)

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


class BridgeStructureEncoding(nn.Module):
    def __init__(self, channels=32, k_neighbors=16, freq_bands=4,
                 min_scale=0.05, max_scale=100.0, grid_size=1.0):
        super().__init__()
        self.channels = channels
        self.k = k_neighbors
        self.freq_bands = freq_bands
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.grid_size = grid_size  # 网格大小，用于绝对位置编码

        # 生成频率
        freqs = 2.0 ** torch.linspace(0., freq_bands - 1, freq_bands)
        self.register_buffer('freqs', freqs)

        # 计算输入特征维度
        self.abs_pos_dim = 6 * freq_bands  # 绝对位置编码维度
        self.rel_pos_dim = 3  # 相对位置维度
        self.local_struct_dim = 13  # 局部结构特征维度
        self.total_dim = self.abs_pos_dim + self.rel_pos_dim + self.local_struct_dim

        # 特征融合的MLP
        self.structure_mlp = nn.Sequential(
            nn.Conv2d(self.total_dim, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1)
        )

    def compute_absolute_position_encoding(self, xyz):
        """
        计算绝对位置编码
        输入:
            xyz: [B, N, 3] 点云坐标
        输出:
            abs_enc: [B, N, 6*freq_bands] 绝对位置编码
        """
        B, N, _ = xyz.shape

        # 1. 网格化坐标（保证位置编码的局部性）
        grid_xyz = torch.floor(xyz / self.grid_size) * self.grid_size

        # 2. 计算每个点的绝对位置编码
        abs_enc = []
        for freq in self.freqs:
            for func in [torch.sin, torch.cos]:
                # 对网格化后的坐标进行编码
                enc = func(grid_xyz * freq)
                abs_enc.append(enc)

        abs_enc = torch.cat(abs_enc, dim=-1)  # [B, N, 6*freq_bands]
        return abs_enc

    def forward(self, xyz):
        B, N, _ = xyz.shape
        device = xyz.device

        # 1. 计算绝对位置编码（这部分对同一个点永远相同）
        abs_pos_enc = self.compute_absolute_position_encoding(xyz)  # [B, N, 6*freq_bands]

        # 2. K近邻搜索
        dist = torch.cdist(xyz, xyz)
        k = min(self.k, N)
        _, idx = dist.topk(k, dim=-1, largest=False)

        idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
        idx = idx + idx_base
        idx = idx.view(-1)

        # 3. 获取邻居点
        neighbors = xyz.view(B * N, -1)[idx].view(B, N, k, -1)
        center = xyz.unsqueeze(2).expand(-1, -1, k, -1)

        # 4. 计算相对位置
        rel_pos = neighbors - center  # [B, N, k, 3]

        # 5. 计算局部结构特征
        structure_features = self.get_structure_features(rel_pos)  # [B, N, 13]

        # 6. 组合特征
        # 扩展绝对位置编码和结构特征到k个邻居
        abs_pos_enc = abs_pos_enc.unsqueeze(2).expand(-1, -1, k, -1)
        structure_features = structure_features.unsqueeze(2).expand(-1, -1, k, -1)

        combined_features = torch.cat([
            abs_pos_enc,  # [B, N, k, 6*freq_bands] 绝对位置编码
            rel_pos,  # [B, N, k, 3] 相对位置
            structure_features  # [B, N, k, 13] 局部结构特征
        ], dim=-1) # [B, N, k, 6*freq_bands + 3 + 13]

        # 7. 通过MLP处理
        combined_features = combined_features.permute(0, 3, 1, 2) # [B, 6*freq_bands + 3 + 13, N, k]
        encoded = self.structure_mlp(combined_features)
        encoded = torch.max(encoded, dim=-1)[0]  # [B, channels, N]

        return encoded

    # get_structure_features方法保持不变

    def get_structure_features(self, rel_pos):
        """
        输入:
            rel_pos: [2, 4096, 16, 3] # [B, N, k, 3]
        输出:
            features: [2, 4096, 13]
        """
        B, N, k, _ = rel_pos.shape

        # 1. 局部主方向特征
        rel_pos_2d = rel_pos.view(B * N, k, 3)
        cov_matrix = torch.bmm(rel_pos_2d.transpose(1, 2), rel_pos_2d) / (k - 1)

        try:
            eigenvalues, _ = torch.linalg.eigh(cov_matrix)
            eigenvalues = eigenvalues.view(B, N, 3)

            linearity = (eigenvalues[..., 0] - eigenvalues[..., 1]) / (eigenvalues[..., 0] + 1e-8)
            planarity = (eigenvalues[..., 1] - eigenvalues[..., 2]) / (eigenvalues[..., 0] + 1e-8)
            sphericity = eigenvalues[..., 2] / (eigenvalues[..., 0] + 1e-8)

            structure_features = torch.stack([linearity, planarity, sphericity], dim=-1)
        except:
            structure_features = torch.zeros((B, N, 3), device=rel_pos.device)

        # 2. 局部统计特征
        center = rel_pos.mean(dim=2, keepdim=True)
        distances = torch.norm(rel_pos - center, dim=-1)

        local_stats = torch.stack([
            distances.max(dim=-1)[0],  # local_radius
            distances.mean(dim=-1),  # mean_dist
            distances.std(dim=-1)  # std_dist
        ], dim=-1)  # [B, N, 3]

        # 3. 方向特征
        normalized_pos = rel_pos / (torch.norm(rel_pos, dim=-1, keepdim=True) + 1e-8)
        direction_sim = torch.bmm(
            normalized_pos.view(B * N, k, 3),
            normalized_pos.view(B * N, k, 3).transpose(1, 2)
        ).view(B, N, k, k)
        direction_consistency = direction_sim.mean(dim=(-1, -2))

        # 4. Z轴特征
        z_stats = torch.stack([
            rel_pos[..., 2].std(dim=-1),  # z_variation
            rel_pos[..., 2].max(dim=-1)[0] - rel_pos[..., 2].min(dim=-1)[0]  # z_range
        ], dim=-1)  # [B, N, 2]

        # 5. 平均相对位置
        mean_rel_pos = rel_pos.mean(dim=2)  # [B, N, 3]

        # 组合所有特征 (确保总共13维)
        features = torch.cat([
            structure_features,  # 3维
            local_stats,  # 3维
            direction_consistency.unsqueeze(-1),  # 1维
            z_stats,  # 2维
            mean_rel_pos,  # 3维
            torch.norm(rel_pos.std(dim=2), dim=-1, keepdim=True)  # 1维
        ], dim=-1)  # 总共13维

        # 验证维度
        assert features.shape[-1] == 13, f"Features should have 13 dimensions, but got {features.shape[-1]}"

        return features


class ColorFeatureExtraction(nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 局部颜色特征提取
        self.color_mlp = nn.Sequential(
            nn.Conv1d(in_channels, 16, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # 颜色注意力机制
        self.color_attention = nn.Sequential(
            nn.Conv1d(32, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 1),
            nn.Sigmoid()
        )

        # 颜色上下文编码
        self.color_context = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(32, 16, 1),
            nn.ReLU(),
            nn.Conv1d(16, 32, 1),
            nn.Sigmoid()
        )

    def forward(self, colors, xyz):
        """
        colors: [B, 3, N]
        xyz: [B, N, 3]
        """
        # 1. 基础颜色特征提取
        color_features = self.color_mlp(colors)  # [B, 32, N]

        # 2. 计算局部颜色相似度
        B, _, N = color_features.shape

        # 找到空间近邻点
        dist = torch.cdist(xyz, xyz)
        k = 16  # k近邻数量
        _, idx = dist.topk(k=k, dim=-1, largest=False)

        # 获取近邻的颜色特征
        idx_base = torch.arange(0, B, device=xyz.device).view(-1, 1, 1) * N
        idx = idx + idx_base
        neighbors_features = color_features.view(B * N, -1)[idx.view(-1)].view(B, N, k, -1)

        # 3. 计算颜色注意力
        color_weights = self.color_attention(color_features)
        enhanced_local = color_features * color_weights

        # 4. 全局颜色上下文
        context_weights = self.color_context(color_features)
        enhanced_global = enhanced_local * context_weights

        return enhanced_global  # [B, 32, N]


class CompositeFeatureFusion(nn.Module):
    def __init__(self, spatial_channels, color_channels):
        super().__init__()
        total_channels = spatial_channels + color_channels
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(total_channels, spatial_channels, 1),
            nn.BatchNorm1d(spatial_channels),
            nn.ReLU()
        )

    def forward(self, spatial_features, color_features):
        # 打印维度以便调试
        #print(f"Spatial features shape: {spatial_features.shape}")
        #print(f"Color features shape: {color_features.shape}")
        fused_features = torch.cat([spatial_features, color_features], dim=1)
        #print(f"Fused features shape: {fused_features.shape}")
        return self.fusion_mlp(fused_features)


if __name__ == '__main__':
    # 测试PositionalEncoding
    xyz = torch.randn(2, 4096, 3) #B, N, C
    # 实例化 EnhancedPositionalEncoding 类
    #pos_enc = EnhancedPositionalEncoding(channels=32, freq_bands=16, k_neighbors=3)

    pos_enc_2 = BridgeStructureEncoding(channels=32, k_neighbors=16, freq_bands=4)

    # 调用 forward 方法
    #output = pos_enc(xyz)

    output_2 = pos_enc_2(xyz)

    # 打印输出形状
    #print(f"Output shape: {output.shape}")
    print(f"Output shape: {output_2.shape}")