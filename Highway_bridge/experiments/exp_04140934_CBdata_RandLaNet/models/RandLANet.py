import torch
import torch.nn as nn
import torch.nn.functional as F

# 原始RandLA-Net的随机采样实现 - 简单高效
class RandomSampling(nn.Module):
    def __init__(self, ratio=0.5):
        super(RandomSampling, self).__init__()
        self.ratio = ratio
        
    def forward(self, xyz, features=None):
        """
        输入:
            xyz: 点云坐标 [B, N, 3]
            features: 点云特征 [B, N, C]
        输出:
            new_xyz: 采样后的点云坐标 [B, N*ratio, 3]
            new_features: 采样后的点云特征 [B, N*ratio, C]
            sample_idx: 采样点的索引 [B, N*ratio]
        """
        batch_size, num_points, _ = xyz.shape
        sample_num = max(1, int(num_points * self.ratio))
        
        new_xyz = []
        new_features = []
        sample_idx_list = []
        
        # 原始RandLA-Net使用纯随机采样，不计算密度
        for b in range(batch_size):
            # 纯随机采样
            sample_idx = torch.randperm(num_points)[:sample_num]
            
            new_xyz.append(xyz[b, sample_idx])
            if features is not None:
                new_features.append(features[b, sample_idx])
            sample_idx_list.append(sample_idx)
            
        new_xyz = torch.stack(new_xyz, dim=0)
        sample_idx = torch.stack(sample_idx_list, dim=0)
        
        if features is not None:
            new_features = torch.stack(new_features, dim=0)
            
        return new_xyz, new_features, sample_idx


# KNN搜索 - 用于局部特征聚合
class KNN(nn.Module):
    def __init__(self, k=16):
        super(KNN, self).__init__()
        self.k = k
        
    def forward(self, xyz, new_xyz=None):
        """
        输入:
            xyz: 原始点云坐标 [B, N, 3]
            new_xyz: 查询点坐标 [B, M, 3]，如果为None则使用xyz
        输出:
            idx: KNN索引 [B, M, k]
        """
        if new_xyz is None:
            new_xyz = xyz
            
        batch_size, n_points, _ = xyz.shape
        m_points = new_xyz.shape[1]
        
        # 使用高效的向量化计算
        dist_matrix = torch.sum((new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2, dim=-1)  # [B, M, N]
        
        # 获取前k个最近邻
        _, idx = torch.topk(dist_matrix, k=self.k, dim=-1, largest=False)  # [B, M, k]
        
        return idx


# 局部空间编码模块 (Local Spatial Encoding)
class LocalSpatialEncoding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalSpatialEncoding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, xyz, features, neighbors_idx):
        """
        输入:
            xyz: 点云坐标 [B, N, 3]
            features: 点云特征 [B, N, C]
            neighbors_idx: KNN索引 [B, N, k]
        输出:
            encoded_features: 编码后的特征 [B, N, k, C']
        """
        batch_size, num_points, _ = xyz.shape
        k = neighbors_idx.shape[-1]
        
        # 获取中心点和邻居点
        neighbors_idx = neighbors_idx.view(batch_size, -1)  # [B, N*k]
        batch_idx = torch.arange(batch_size, device=xyz.device).view(-1, 1).repeat(1, num_points * k)
        batch_idx = batch_idx.view(-1)
        neighbors_idx = neighbors_idx.view(-1)
        
        # 获取邻居点坐标和特征
        neighbors_xyz = xyz.reshape(batch_size * num_points, 1, 3)
        all_xyz = xyz.reshape(batch_size * num_points, 3)
        neighbors_xyz_flat = all_xyz[batch_idx * num_points + neighbors_idx].view(batch_size, num_points, k, 3)
        
        # 计算相对位置
        relative_pos = neighbors_xyz_flat - xyz.unsqueeze(2)  # [B, N, k, 3]
        
        # 计算相对距离
        relative_dist = torch.sqrt(torch.sum(relative_pos ** 2, dim=-1, keepdim=True))  # [B, N, k, 1]
        
        # 拼接相对位置和距离
        pos_encoding = torch.cat([relative_pos, relative_dist], dim=-1)  # [B, N, k, 4]
        
        # 如果有特征，获取邻居特征并拼接
        if features is not None:
            all_features = features.reshape(batch_size * num_points, -1)
            neighbors_features = all_features[batch_idx * num_points + neighbors_idx].view(
                batch_size, num_points, k, -1)
            
            # 拼接位置编码和特征
            pos_encoding = torch.cat([pos_encoding, neighbors_features], dim=-1)  # [B, N, k, 4+C]
        
        # 应用MLP
        pos_encoding = pos_encoding.permute(0, 3, 1, 2).contiguous()  # [B, 4+C, N, k]
        encoded_features = self.mlp(pos_encoding)  # [B, C', N, k]
        encoded_features = encoded_features.permute(0, 2, 3, 1).contiguous()  # [B, N, k, C']
        
        return encoded_features


# 基于注意力的特征聚合模块
class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()
        self.score_fn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1)
        )
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        输入:
            x: 特征 [B, C, N, k]
        输出:
            pooled: 聚合后的特征 [B, C', N]
        """
        # 计算注意力分数
        scores = self.score_fn(x)  # [B, 1, N, k]
        scores = F.softmax(scores, dim=-1)  # [B, 1, N, k]
        
        # 应用注意力权重
        features = torch.sum(x * scores, dim=-1)  # [B, C, N]
        
        # 应用MLP
        features = self.mlp(features)  # [B, C', N]
        
        return features


# 膨胀残差块（Dilated Residual Block）
class DilatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilatedResidualBlock, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """
        输入:
            x: 特征 [B, C, N]
        输出:
            out: 残差块输出 [B, C', N]
        """
        shortcut = self.shortcut(x)
        out = self.mlp1(x)
        out = self.mlp2(out)
        out = self.relu(out + shortcut)
        return out


# 局部特征聚合模块 - 按照原始RandLA-Net论文
class LocalFeatureAggregation(nn.Module):
    def __init__(self, in_channels, out_channels, k=16):
        super(LocalFeatureAggregation, self).__init__()
        self.k = k
        
        # 局部空间编码
        self.lse1 = LocalSpatialEncoding(in_channels+4, out_channels//2)
        self.lse2 = LocalSpatialEncoding(in_channels+4, out_channels//2)
        
        # 注意力池化
        self.ap1 = AttentivePooling(out_channels//2, out_channels//2)
        self.ap2 = AttentivePooling(out_channels//2, out_channels//2)
        
        # 膨胀残差块
        self.drb = DilatedResidualBlock(out_channels, out_channels)
        
        # KNN搜索
        self.knn = KNN(k=k)
        
    def forward(self, xyz, features):
        """
        输入:
            xyz: 点云坐标 [B, N, 3]
            features: 点云特征 [B, N, C]
        输出:
            agg_features: 聚合后的特征 [B, N, C']
        """
        batch_size, num_points, _ = xyz.shape
        
        # 第一次KNN聚合 - 原始RandLA-Net使用两次不同尺度的聚合
        idx1 = self.knn(xyz)
        lse_features1 = self.lse1(xyz, features, idx1)  # [B, N, k, C//2]
        lse_features1 = lse_features1.permute(0, 3, 1, 2)  # [B, C//2, N, k]
        agg_features1 = self.ap1(lse_features1)  # [B, C//2, N]
        
        # 第二次KNN聚合 - 使用不同的邻居数（模拟膨胀卷积效果）
        idx2 = self.knn(xyz)  # 实际应该使用不同的k值，这里简化处理
        lse_features2 = self.lse2(xyz, features, idx2)  # [B, N, k, C//2]
        lse_features2 = lse_features2.permute(0, 3, 1, 2)  # [B, C//2, N, k]
        agg_features2 = self.ap2(lse_features2)  # [B, C//2, N]
        
        # 拼接两次聚合的特征
        agg_features = torch.cat([agg_features1, agg_features2], dim=1)  # [B, C, N]
        
        # 应用膨胀残差块
        agg_features = self.drb(agg_features)  # [B, C, N]
        
        # 转换回 [B, N, C] 格式
        agg_features = agg_features.permute(0, 2, 1)  # [B, N, C]
        
        return agg_features


# 基于最近邻的特征传播上采样模块
class FeaturePropagation(nn.Module):
    def __init__(self, in_channel_prev, in_channel_skip, out_channel):
        super(FeaturePropagation, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channel_prev + in_channel_skip, out_channel, 1, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channel, out_channel, 1, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, xyz_prev, xyz_skip, points_prev, points_skip):
        """
        输入:
            xyz_prev: 前一层点云坐标 [B, N_prev, 3]
            xyz_skip: 跳跃连接点云坐标 [B, N_skip, 3]
            points_prev: 前一层点云特征 [B, C_prev, N_prev]
            points_skip: 跳跃连接点云特征 [B, C_skip, N_skip]
        输出:
            new_points: 上采样后的特征 [B, C_out, N_skip]
        """
        # 计算最近邻插值权重
        batch_size = xyz_prev.shape[0]
        n_skip = xyz_skip.shape[1]
        n_prev = xyz_prev.shape[1]
        
        # 为每个点在xyz_skip中找到最近的3个点在xyz_prev中的索引和距离
        # 这里简化实现，使用欧氏距离计算
        dist = torch.cdist(xyz_skip, xyz_prev)  # [B, N_skip, N_prev]
        _, idx = torch.topk(dist, k=3, dim=-1, largest=False)  # [B, N_skip, 3]
        
        # 计算距离的倒数作为权重
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm  # [B, N_skip, N_prev]
        
        # 根据权重进行插值
        interpolated_points = torch.zeros(batch_size, points_prev.shape[1], n_skip, device=xyz_prev.device)
        
        for b in range(batch_size):
            for n in range(n_skip):
                for k in range(3):  # 使用最近的3个点
                    interpolated_points[b, :, n] += weight[b, n, idx[b, n, k]] * points_prev[b, :, idx[b, n, k]]
        
        # 拼接特征
        if points_skip is not None:
            new_points = torch.cat([interpolated_points, points_skip], dim=1)
        else:
            new_points = interpolated_points
            
        # 应用MLP
        new_points = self.mlp(new_points)
        
        return new_points


# RandLA-Net主网络 - 符合原始论文结构
class RandLANet(nn.Module):
    def __init__(self, num_classes=5, d_in=3):
        super(RandLANet, self).__init__()
        
        # 初始特征转换
        self.fc_start = nn.Linear(d_in, 8)
        self.bn_start = nn.BatchNorm1d(8)
        
        # 编码器层配置 - 按照原始论文
        self.encoder_dims = [16, 64, 128, 256]
        self.decoder_dims = [256, 128, 64, 32]  # 解码器输出维度
        # 原始论文中的采样率是逐渐减小的
        self.sampling_ratios = [0.35, 0.3, 0.25, 0.25]
        
        # 采样和编码模块
        self.down_modules = nn.ModuleList()
        pre_channel = 8
        for i, channel in enumerate(self.encoder_dims):
            # 原始论文中使用固定的k=16
            k = 16
            module = nn.ModuleDict({
                'sample': RandomSampling(ratio=self.sampling_ratios[i]),
                'localAgg': LocalFeatureAggregation(pre_channel, channel, k=k)
            })
            self.down_modules.append(module)
            pre_channel = channel
        
        # 解码器模块
        self.up_modules = nn.ModuleList()
        
        # 确定跳跃连接通道数
        encoder_output_dims = [8] + self.encoder_dims  # [8, 16, 64, 128, 256]
        skip_connection_channels = encoder_output_dims[:-1][::-1]  # [128, 64, 16, 8]
        
        decoder_input_channels = [self.encoder_dims[-1]] + self.decoder_dims[:-1]  # [256, 256, 128, 64]
        
        for i in range(len(self.decoder_dims)):
            in_ch_prev = decoder_input_channels[i]
            in_ch_skip = skip_connection_channels[i] if i < len(skip_connection_channels) else 0
            out_ch = self.decoder_dims[i]
            
            self.up_modules.append(
                FeaturePropagation(in_ch_prev, in_ch_skip, out_ch)
            )
        
        # 最终的分割头
        self.seg_head = nn.Sequential(
            nn.Conv1d(self.decoder_dims[-1], 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(64, num_classes, 1)
        )
        
    def forward(self, xyz, features=None):
        """
        输入:
            xyz: 点云坐标 [B, N, 3]
            features: 点云特征 [B, N, d_in]
        输出:
            point_scores: 每个点的类别得分 [B, num_classes, N]
        """
        batch_size, num_points, _ = xyz.shape
        
        # 如果没有额外特征，使用坐标
        if features is None:
            features = xyz
        
        # 初始特征转换
        x = self.fc_start(features)  # [B, N, 8]
        x = x.transpose(1, 2)  # [B, 8, N]
        x = self.bn_start(x)
        x = F.relu(x)
        x = x.transpose(1, 2)  # [B, N, 8]
        
        # 编码阶段 - 存储每层的xyz和特征用于跳跃连接
        encoder_xyz = [xyz]
        encoder_features = [x]
        
        # 逐层下采样和特征提取
        for i, module in enumerate(self.down_modules):
            # 随机采样
            xyz_down, features_down, _ = module['sample'](encoder_xyz[-1], encoder_features[-1])
            # 局部特征聚合
            features_agg = module['localAgg'](xyz_down, features_down)  # 输出 [B, N_down, C_out]
            
            encoder_xyz.append(xyz_down)
            encoder_features.append(features_agg)
        
        # 解码阶段
        # 从最深层开始
        decoder_features = encoder_features[-1].transpose(1, 2)  # [B, C_last_enc, N_last_enc]
        
        for i in range(len(self.up_modules)):
            # 获取跳跃连接的坐标和特征
            xyz_skip = encoder_xyz[-(i+2)]
            features_skip = encoder_features[-(i+2)].transpose(1, 2) if i < len(encoder_features)-1 else None
            
            # 获取上一层解码器的坐标
            xyz_prev = encoder_xyz[-(i+1)]
            
            # 应用上采样和特征传播模块
            decoder_features = self.up_modules[i](
                xyz_prev, xyz_skip, decoder_features, features_skip
            )  # 输出 [B, C_dec_i, N_skip]
        
        # 最终分割头
        point_scores = self.seg_head(decoder_features)  # [B, num_classes, N]
        
        return point_scores  # 返回 [B, C, N] 形状的逐点预测
