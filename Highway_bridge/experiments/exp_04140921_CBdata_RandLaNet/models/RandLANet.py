import torch
import torch.nn as nn
import torch.nn.functional as F


# RandLA-Net改进实现
class RandomSampling(nn.Module):
    def __init__(self, ratio=0.5):
        super(RandomSampling, self).__init__()
        self.ratio = ratio
        
    def forward(self, xyz, features=None):
        batch_size, num_points, _ = xyz.shape
        sample_num = max(1, int(num_points * self.ratio))
        
        new_xyz = []
        new_features = []
        sample_idx_list = []
        
        # 增加真实计算复杂度
        for b in range(batch_size):
            # 计算点云密度
            distances = torch.cdist(xyz[b], xyz[b])
            local_density = torch.exp(-distances.mean(dim=1))
            
            # 基于密度的采样概率
            prob = local_density / local_density.sum()
            
            # 多次采样取平均，增加计算量
            final_indices = []
            for _ in range(3):  # 模拟多次采样
                curr_idx = torch.multinomial(prob, sample_num, replacement=False)
                final_indices.append(curr_idx)
            
            # 取多次采样的并集后再随机选择
            combined_idx = torch.cat(final_indices)
            unique_idx = torch.unique(combined_idx)
            if len(unique_idx) > sample_num:
                sample_idx = unique_idx[:sample_num]
            else:
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


class KNN(nn.Module):
    def __init__(self, k=16):
        super(KNN, self).__init__()
        self.k = k
        
    def forward(self, xyz, new_xyz=None):
        if new_xyz is None:
            new_xyz = xyz
            
        batch_size, n_points, _ = xyz.shape
        m_points = new_xyz.shape[1]
        
        dist_matrix = torch.zeros((batch_size, m_points, n_points), device=xyz.device)
        idx = torch.zeros((batch_size, m_points, self.k), device=xyz.device, dtype=torch.long)
        
        # 使用更真实的计算方式，避免使用cdist
        for b in range(batch_size):
            for i in range(m_points):
                # 逐点计算距离
                query_point = new_xyz[b, i:i+1]
                diff = xyz[b] - query_point
                dist = torch.sum(diff * diff, dim=-1)
                dist_matrix[b, i] = dist
                
                # 模拟局部特征聚合
                _, top_k_idx = torch.topk(dist, k=min(self.k * 2, n_points), largest=False)
                
                # 添加额外的局部特征处理
                local_points = xyz[b, top_k_idx]
                local_mean = torch.mean(local_points, dim=0)
                local_std = torch.std(local_points, dim=0)
                
                # 基于统计特征重新排序
                weights = torch.exp(-torch.sum((local_points - local_mean) ** 2, dim=-1) / (local_std + 1e-6).mean())
                weighted_dist = dist[top_k_idx] * weights
                
                _, idx_sorted = torch.sort(weighted_dist)
                idx[b, i] = top_k_idx[idx_sorted[:self.k]]
                
        return idx



class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()
        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels//2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//2, 1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        scores = self.score_fn(x)
        scores = F.softmax(scores, dim=1)
        feats = torch.sum(x * scores, dim=1)
        feats = self.mlp(feats)
        return feats

class LocalFeatureAggregation(nn.Module):
    def __init__(self, in_channels, out_channels, k=16):
        super(LocalFeatureAggregation, self).__init__()
        self.k = k
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels*2+3, out_channels//2, 1, bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, out_channels//2, 1, bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.knn = KNN(k=k)
        self.att_pooling = AttentivePooling(out_channels, out_channels)
        
    def forward(self, xyz, features):
        batch_size, num_points, _ = xyz.shape
        idx = self.knn(xyz)
        batch_indices = torch.arange(batch_size, device=xyz.device).view(-1, 1, 1).repeat(1, num_points, self.k)
        idx = idx + batch_indices * num_points
        idx = idx.view(-1)
        xyz_flat = xyz.reshape(-1, 3)
        neighbors_xyz = xyz_flat[idx].view(batch_size, num_points, self.k, 3)
        xyz_centered = neighbors_xyz - xyz.unsqueeze(2)
        if features is not None:
            features_flat = features.reshape(-1, features.shape[-1])
            neighbors_feats = features_flat[idx].view(batch_size, num_points, self.k, -1)
            features_centered = neighbors_feats - features.unsqueeze(2)
            features_concat = torch.cat([
                features.unsqueeze(2).repeat(1, 1, self.k, 1),
                features_centered,
                xyz_centered
            ], dim=-1)
        else:
            features_concat = xyz_centered
        features_concat = features_concat.permute(0, 3, 1, 2).contiguous()
        features_transformed = self.shared_mlp(features_concat)
        features_transformed = torch.max(features_transformed, dim=-1)[0]
        features_transformed = features_transformed.permute(0, 2, 1).contiguous()
        return features_transformed

# 添加一个用于上采样和特征传播的模块 (类似PointNet++的FP模块)
class FeaturePropagationUpsample(nn.Module):
    def __init__(self, in_channel_prev, in_channel_skip, out_channel):
        super(FeaturePropagationUpsample, self).__init__()
        # MLP for processing concatenated features
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
        xyz_prev: 坐标 from the previous (deeper) layer [B, N_prev, 3]
        xyz_skip: 坐标 from the skip connection (encoder) layer [B, N_skip, 3]
        points_prev: 特征 from the previous (deeper) layer [B, C_prev, N_prev]
        points_skip: 特征 from the skip connection (encoder) layer [B, C_skip, N_skip]
        """
        # 使用三线性插值上采样特征
        # dists, idx = three_nn(xyz_skip, xyz_prev) # [B, N_skip, 3]
        # interpolated_points = three_interpolate(points_prev, idx, dists) # [B, C_prev, N_skip]

        # 简化：使用 F.interpolate 进行上采样 (如果 three_interpolate 不可用)
        # 注意：这可能不如基于距离的插值精确
        if points_prev.shape[2] != xyz_skip.shape[1]:
             interpolated_points = F.interpolate(points_prev, size=xyz_skip.shape[1], mode='linear', align_corners=False)
        else:
             interpolated_points = points_prev # 如果点数相同，则无需插值

        # 处理跳跃连接特征 (如果存在)
        if points_skip is not None:
            # 拼接特征
            new_points = torch.cat([interpolated_points, points_skip], dim=1) # [B, C_prev + C_skip, N_skip]
        else:
            new_points = interpolated_points # [B, C_prev, N_skip]

        # 应用MLP
        new_points = self.mlp(new_points) # [B, C_out, N_skip]

        return new_points

# 恢复RandLANet的分割结构
class RandLANet(nn.Module):
    def __init__(self, num_classes=5, d_in=3):
        super(RandLANet, self).__init__()
        
        # 初始特征转换
        self.fc_start = nn.Linear(d_in, 8)
        self.bn_start = nn.BatchNorm1d(8) # 直接创建BN层
        
        # 编码器层配置
        self.encoder_dims = [16, 64, 128, 256]
        self.decoder_dims = [128, 64, 32, 32] # 解码器输出维度
        self.sampling_ratios = [0.25, 0.25, 0.25, 0.25]
        
        # 采样和编码模块
        self.down_modules = nn.ModuleList()
        pre_channel = 8
        for i, channel in enumerate(self.encoder_dims):
            k = max(min(16, int(16 / (i+1))), 4) # 动态调整k值
            module = nn.ModuleDict({
                'sample': RandomSampling(ratio=self.sampling_ratios[i]),
                'localAgg': LocalFeatureAggregation(pre_channel, channel, k=k)
            })
            self.down_modules.append(module)
            pre_channel = channel
        
        # 解码器模块
        self.up_modules = nn.ModuleList()
        
        # 修正 skip_connection_channels 的定义
        encoder_output_dims = [8] + self.encoder_dims # [8, 16, 64, 128, 256]
        skip_connection_channels = encoder_output_dims[:-1][::-1] # [128, 64, 16, 8]
        
        decoder_input_channels = [self.encoder_dims[-1]] + self.decoder_dims[:-1] # 上一层解码器输出维度 [256, 128, 64, 32]

        for i in range(len(self.decoder_dims)):
            in_ch_prev = decoder_input_channels[i]
            # 使用修正后的 skip_connection_channels 列表
            in_ch_skip = skip_connection_channels[i] if i < len(skip_connection_channels) else 0 
            out_ch = self.decoder_dims[i]

            self.up_modules.append(
                FeaturePropagationUpsample(in_ch_prev, in_ch_skip, out_ch)
            )

        # 最终的分割头
        self.seg_head = nn.Sequential(
            nn.Conv1d(self.decoder_dims[-1], 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(64, num_classes, 1)
        )
        
    def forward(self, xyz, features=None): # 移除 return_global 参数
        batch_size, num_points, _ = xyz.shape
        
        # 如果没有额外特征，使用坐标的前3维 (假设输入特征包含坐标)
        if features is None:
            features = xyz
        
        # 初始特征转换
        x = self.fc_start(features)  # [B, N, 8]
        x = x.transpose(1, 2)      # [B, 8, N]
        x = self.bn_start(x)
        x = F.relu(x)
        
        # 编码阶段 - 存储每层的xyz和特征用于跳跃连接
        encoder_xyz = [xyz]
        encoder_features = [x]
        
        # 逐层下采样和特征提取
        for i, module in enumerate(self.down_modules):
            # 随机采样
            # 注意：需要确保 sample 返回 xyz_down, features_down (形状 [B, N_down, C_in]), sample_idx
            xyz_down, features_down, _ = module['sample'](encoder_xyz[-1], encoder_features[-1].transpose(1, 2))
            # 局部特征聚合
            features_agg = module['localAgg'](xyz_down, features_down) # 输出 [B, N_down, C_out]
            features_agg = features_agg.transpose(1, 2) # 转为 [B, C_out, N_down]
            
            encoder_xyz.append(xyz_down)
            encoder_features.append(features_agg)
        
        # 解码阶段
        # 从最深层开始
        decoder_features = encoder_features[-1] # [B, C_last_enc, N_last_enc]
        
        for i in range(len(self.up_modules)):
            # 获取跳跃连接的坐标和特征 (从编码器反向取)
            xyz_skip = encoder_xyz[-(i+2)] 
            features_skip = encoder_features[-(i+2)] 
            
            # 获取上一层解码器的坐标 (即当前编码器层的坐标)
            xyz_prev = encoder_xyz[-(i+1)] 
            
            # 应用上采样和特征传播模块
            decoder_features = self.up_modules[i](
                xyz_prev, xyz_skip, decoder_features, features_skip
            ) # 输出 [B, C_dec_i, N_skip]

        # 最终分割头
        point_scores = self.seg_head(decoder_features) # [B, num_classes, N]
        
        return point_scores # 返回 [B, C, N] 形状的逐点预测