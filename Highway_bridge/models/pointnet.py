import torch
import torch.nn as nn
import torch.nn.functional as F

# PointNet基础模块
import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        
        # 3D空间变换网络结构，与论文一致
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        # 初始化最后一层权重为0，偏置为单位矩阵
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()
        self.fc3.bias.data[:k*k:k+1] = 1.0

    def forward(self, x):
        batch_size = x.size()[0]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # 重塑为变换矩阵
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, self.k*self.k).repeat(batch_size, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        
        return x

class PointNetSeg(nn.Module):
    def __init__(self, num_classes=5, feature_transform=True, feature_dim=3):
        super(PointNetSeg, self).__init__()
        self.feature_transform = feature_transform
        self.feature_dim = feature_dim
        
        # 输入变换网络 (只对xyz坐标进行变换)
        self.input_transform = TNet(k=3)
        # 特征变换网络（如果启用）
        self.feature_transform_net = TNet(k=64) if feature_transform else None
        
        # 修改第一层卷积的输入通道数为3+feature_dim
        input_channels = 3 + feature_dim
        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        
        self.seg_conv1 = nn.Conv1d(1088, 512, 1)
        self.seg_conv2 = nn.Conv1d(512, 256, 1)
        self.seg_conv3 = nn.Conv1d(256, 128, 1)
        self.seg_conv4 = nn.Conv1d(128, num_classes, 1)
        
        self.bn_seg1 = nn.BatchNorm1d(512)
        self.bn_seg2 = nn.BatchNorm1d(256)
        self.bn_seg3 = nn.BatchNorm1d(128)
        
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, xyz, features):
        # 确保输入正确
        assert xyz is not None, "xyz coordinates cannot be None"
        assert features is not None, "features cannot be None"
        assert xyz.dim() == 3, "xyz should be [B, N, 3]"
        assert features.dim() == 3, "features should be [B, N, feature_dim]"
        assert features.size(2) == self.feature_dim, f"features should have {self.feature_dim} channels"
        
        B, N = xyz.size(0), xyz.size(1)
        
        # 转置xyz为[B, 3, N]用于输入变换
        xyz_t = xyz.transpose(2, 1)
        
        # 应用输入变换（仅对xyz）
        trans_input = self.input_transform(xyz_t)
        xyz = torch.bmm(xyz, trans_input.transpose(2, 1))
        
        # 准备网络输入：将变换后的xyz和features拼接
        # 先将两者都转置为[B, C, N]格式
        xyz_t = xyz.transpose(2, 1)  # [B, 3, N]
        features_t = features.transpose(2, 1)  # [B, feature_dim, N]
        
        # 拼接特征
        x = torch.cat([xyz_t, features_t], dim=1)  # [B, 3+feature_dim, N]
        
        # 特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 特征变换
        if self.feature_transform_net is not None:
            trans_feat = self.feature_transform_net(x)
            x = torch.bmm(trans_feat, x)
        else:
            trans_feat = None
            
        point_feat = x
        
        # 全局特征
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        global_feat = torch.max(x, 2, keepdim=True)[0]
        global_feat = global_feat.repeat(1, 1, N)
        
        # 拼接局部和全局特征
        x = torch.cat([point_feat, global_feat], dim=1)
        
        # 分割头
        x = F.relu(self.bn_seg1(self.seg_conv1(x)))
        x = F.relu(self.bn_seg2(self.seg_conv2(x)))
        x = F.relu(self.bn_seg3(self.seg_conv3(x)))
        x = self.dropout(x)
        x = self.seg_conv4(x)
        
        x = x.transpose(2, 1).contiguous()
        
        return x

# 添加到PointNetSeg类中
def get_feature_transform_regularizer(self):
    """计算特征变换矩阵的正则化损失"""
    if not hasattr(self, 'trans_feat') or self.trans_feat is None:
        return 0.0
    
    d = self.trans_feat.size()[1]
    I = torch.eye(d)[None, :, :].to(self.trans_feat.device)
    loss = torch.mean(torch.norm(torch.bmm(self.trans_feat, self.trans_feat.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

