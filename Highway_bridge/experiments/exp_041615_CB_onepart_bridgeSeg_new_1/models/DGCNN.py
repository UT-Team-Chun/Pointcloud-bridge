import torch
import torch.nn as nn
import torch.nn.functional as F

# DGCNN模型实现
class DGCNN(nn.Module):
    def __init__(self, num_classes=5, k=64):
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
        batch_size, dim, num_points = x.size()
        x = x.transpose(2, 1)  # 转换为 [B, N, D]
        
        # 计算成对距离矩阵
        dist_matrix = torch.zeros((batch_size, num_points, num_points), device=x.device)
        for b in range(batch_size):
            # 逐点计算距离，避免过大的中间张量
            for i in range(num_points):
                # 计算当前点与所有点之间的距离
                diff = x[b, i:i+1] - x[b]  # [1, N, D] - [N, D] = [N, D]
                dist_matrix[b, i] = torch.sum(diff * diff, dim=-1)  # [N]
        
        # 获取前k个最近邻
        _, idx = torch.topk(-dist_matrix, k=k, dim=-1)  # 使用负距离找最近的点
        return idx

    def get_graph_feature(self, x, k=20, idx=None):
        batch_size, num_dims, num_points = x.size()
        
        if idx is None:
            idx = self.knn(x, k)
        
        # 优化特征构建方式，减少循环层数
        feature = torch.zeros(batch_size, num_points, k, num_dims*2, device=x.device)
        
        for b in range(batch_size):
            # 获取所有中心点特征 [N, D]
            center_feats = x[b].transpose(0, 1)  # [N, D]
            
            for i in range(num_points):
                center_feat = center_feats[i]  # [D]
                # 获取k个邻居的索引
                neighbor_indices = idx[b, i]  # [k]
                # 批量获取邻居特征 [k, D]
                neighbor_feats = torch.index_select(center_feats, 0, neighbor_indices)
                
                # 计算差异特征并拼接
                diff_feats = neighbor_feats - center_feat.unsqueeze(0)  # [k, D]
                # 拼接差异和中心特征
                feature[b, i, :, :num_dims] = diff_feats
                feature[b, i, :, num_dims:] = center_feat.unsqueeze(0).expand(k, -1)
        
        # 调整维度顺序
        feature = feature.permute(0, 3, 1, 2)
        
        return feature

    def forward(self, xyz, features=None):
        B, N, C = xyz.shape
        
        # 如果提供了特征，则与坐标拼接
        if features is not None:
            xyz = torch.cat([xyz, features], dim=2)
            
        x = xyz.transpose(2, 1)
        
        batch_size = x.size(0)
        x = x[:, :3, :]  # 只使用坐标信息
        
        # EdgeConv模块
        x1 = self.get_graph_feature(x, k=self.k)
        x1 = self.conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]
        
        x2 = self.get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]
        
        x3 = self.get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x3)
        x3 = x3.max(dim=-1, keepdim=False)[0]
        
        x4 = self.get_graph_feature(x3, k=self.k)
        x4 = self.conv4(x4)
        x4 = x4.max(dim=-1, keepdim=False)[0]
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        x = self.conv5(x)
        
        # 全局特征
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
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