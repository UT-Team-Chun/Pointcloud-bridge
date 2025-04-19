import torch
import torch.nn as nn
import torch.nn.functional as F

# DGCNN模型实现
class DGCNN(nn.Module):
    def __init__(self, num_classes=5, k=20):
        super(DGCNN, self).__init__()
        self.k = k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm1d(1024)

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
        
        self.linear1 = nn.Linear(1024*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_classes)

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
        
        # EdgeConv模块
        x1 = self.get_graph_feature(x, k=k)
        x1 = self.conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]
        
        x2 = self.get_graph_feature(x1, k=k)
        x2 = self.conv2(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]
        
        x3 = self.get_graph_feature(x2, k=k)
        x3 = self.conv3(x3)
        x3 = x3.max(dim=-1, keepdim=False)[0]
        
        x4 = self.get_graph_feature(x3, k=k)
        x4 = self.conv4(x4)
        x4 = x4.max(dim=-1, keepdim=False)[0]
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        x = self.conv5(x)
        
        # 全局特征
        x1 = F.adaptive_max_pool1d(x, 1).view(B, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(B, -1)
        x = torch.cat((x1, x2), 1)
        
        # MLP分类器
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        
        # 将输出扩展为[B, N, C]形式
        x = x.unsqueeze(1).repeat(1, N, 1)
        
        return x