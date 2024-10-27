# models/attention_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.Conv1d(in_channels, in_channels//4, 1),
            nn.BatchNorm1d(in_channels//4),
            nn.ReLU(),
            nn.Conv1d(in_channels//4, in_channels, 1),
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

