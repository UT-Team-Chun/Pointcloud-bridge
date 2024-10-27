# models/enhanced_pointnet2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import EnhancedSetAbstraction, FeaturePropagation
from .attention_modules import PositionalEncoding, BoundaryAwareModule

class EnhancedPointNet2(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.pos_encoding = PositionalEncoding(64)
        
        # Encoder
        self.sa1 = EnhancedSetAbstraction(1024, 0.1, 32, 6+64, [64, 64, 128])
        self.sa2 = EnhancedSetAbstraction(256, 0.2, 32, 131, [128, 128, 256])
        self.sa3 = EnhancedSetAbstraction(64, 0.4, 32, 259, [256, 256, 512])
        
        # Boundary aware modules
        self.boundary1 = BoundaryAwareModule(128)
        self.boundary2 = BoundaryAwareModule(256)
        self.boundary3 = BoundaryAwareModule(512)
        
        # Decoder
        # 解码器部分调整
        self.fp3 = FeaturePropagation(768, [256, 256])  # 修改为768 (512 + 256)
        self.fp2 = FeaturePropagation(384, [256, 128])  # 256 + 128 = 387
        self.fp1 = FeaturePropagation(128, [128, 128, 128])  # 128 + 3 = 131
        
        # Final layers
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
    
    def forward(self, xyz, points):
        """
        xyz: [B, N, 3]
        points: [B, N, 3] (RGB)
        """
        # Add positional encoding
        pos_enc = self.pos_encoding(xyz)  # [B, 64, N]
        points = torch.cat([points.transpose(1,2), pos_enc], dim=1)  # [B, 67, N]
        
        
        # Encoder
        l1_xyz, l1_points = self.sa1(xyz, points)
        l1_points = self.boundary1(l1_points, l1_xyz)
        
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.boundary2(l2_points, l2_xyz)
        
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.boundary3(l3_points, l3_xyz)
        
        # Decoder
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)
        
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        feat = self.drop1(feat)
        out = self.conv2(feat)
        
        return out