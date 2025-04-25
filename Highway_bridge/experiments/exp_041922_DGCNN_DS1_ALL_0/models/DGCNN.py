import torch
import torch.nn as nn
import torch.nn.functional as F

# DGCNN模型实现 - 改进版，更适合点云分割
class DGCNN(nn.Module):
    def __init__(self, num_classes=5, k=20):
        super(DGCNN, self).__init__()
        self.k = k
        
        # 特征提取网络
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        # 边缘卷积层
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(320, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        # 新增: 局部特征处理
        self.local_bn = nn.BatchNorm1d(320)
        
        # 新增: 点级分类层
        self.point_conv = nn.Sequential(
            nn.Conv1d(1344, 512, 1),                      # 1344 = 320(局部) + 1024(全局)
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, num_classes, 1)
        )
        
    def knn(self, x, k):
        """
        高效实现的KNN搜索，使用向量化操作代替嵌套循环
        输入:
            x: 点云特征 [B, D, N]
        输出:
            idx: KNN索引 [B, N, k]
        """
        batch_size, dim, num_points = x.size()
        
        # 转置为 [B, N, D]
        x = x.transpose(2, 1).contiguous()
        
        # 计算欧几里得距离矩阵 - 向量化实现
        inner = -2 * torch.matmul(x, x.transpose(2, 1))  # [B, N, N]
        xx = torch.sum(x**2, dim=2, keepdim=True)  # [B, N, 1]
        pairwise_distance = xx + inner + xx.transpose(2, 1)  # [B, N, N]
        
        # 获取 k 个最近邻的索引
        _, idx = torch.topk(-pairwise_distance, k=k, dim=-1)  # [B, N, k]
        
        return idx

    def get_graph_feature(self, x, k=20, idx=None):
        """
        高效实现的图特征提取，通过索引操作和转置代替嵌套循环
        输入:
            x: 点云特征 [B, D, N]
            k: 邻居数量
            idx: 预计算的邻居索引 [B, N, k]
        输出:
            feature: 图特征 [B, 2D, N, k]
        """
        batch_size, num_dims, num_points = x.size()
        
        # 如果没有提供索引，计算k近邻
        if idx is None:
            idx = self.knn(x, k)  # [B, N, k]
        
        # 准备批处理索引
        device = x.device
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base  # [B, N, k]
        idx = idx.view(-1)  # 展平索引
        
        # 转置特征为 [B, N, D]
        x = x.transpose(2, 1).contiguous()
        
        # 将特征展平为 [B*N, D]
        feature = x.view(batch_size * num_points, -1)[idx, :]
        
        # 重塑特征为 [B, N, k, D]
        feature = feature.view(batch_size, num_points, k, num_dims)
        
        # 获取中心点特征 [B, N, 1, D]
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        
        # 构建最终特征 [B, N, k, 2D]
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        
        return feature

    def forward(self, xyz, features=None):
        """
        输入:
            xyz: 点云坐标 [B, N, 3]
            features: 点云特征 [B, N, C]，可选
        输出:
            x: 分类/分割结果 [B, N, num_classes]
        """
        B, N, C = xyz.shape
        
        # 如果提供了特征，则与坐标拼接
        if features is not None:
            xyz = torch.cat([xyz, features], dim=2)
            
        x = xyz.transpose(2, 1)  # [B, C, N]
        
        # 仅使用坐标信息进行处理
        x = x[:, :3, :]
        
        # 降低k值以提高效率
        k = min(self.k, N-1)  # 确保k不超过点数-1
        
        # EdgeConv模块 - 保留每个层级的局部特征
        x1 = self.get_graph_feature(x, k=k)
        x1 = self.conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]  # [B, 64, N]
        
        x2 = self.get_graph_feature(x1, k=k)
        x2 = self.conv2(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]  # [B, 64, N]
        
        x3 = self.get_graph_feature(x2, k=k)
        x3 = self.conv3(x3)
        x3 = x3.max(dim=-1, keepdim=False)[0]  # [B, 64, N]
        
        x4 = self.get_graph_feature(x3, k=k)
        x4 = self.conv4(x4)
        x4 = x4.max(dim=-1, keepdim=False)[0]  # [B, 128, N]
        
        # 拼接多尺度局部特征
        local_feat = torch.cat((x1, x2, x3, x4), dim=1)  # [B, 320, N]
        
        # 保存局部特征用于后续点级分类
        local_feat_normalized = F.leaky_relu(self.local_bn(local_feat), negative_slope=0.2)
        
        # 处理全局特征 (用于捕获整体结构)
        x = self.conv5(local_feat)  # [B, 1024, N]
        
        # 全局特征池化
        global_feat = F.adaptive_max_pool1d(x, 1)  # [B, 1024, 1]
        
        # 特征融合: 将全局特征与每个点的局部特征拼接
        global_feat_expanded = global_feat.expand(-1, -1, N)  # [B, 1024, N]
        point_features = torch.cat([local_feat_normalized, global_feat_expanded], dim=1)  # [B, 1344, N]
        
        # 点级分类
        logits = self.point_conv(point_features)  # [B, num_classes, N]
        
        # 转换为 [B, N, num_classes] 格式以符合接口要求
        logits = logits.transpose(1, 2)  # [B, N, num_classes]
        
        return logits