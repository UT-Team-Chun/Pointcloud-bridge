# models/enhanced_pointnet2.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_modules import BoundaryAwareModule, EnhancedAttentionModule, \
    GeometricFeatureExtraction, EnhancedPositionalEncoding
from .pointnet2_utils import FeaturePropagation, SetAbstraction


class EnhancedPointNet2(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        input_ch=6
        self.pos_encoding = EnhancedPositionalEncoding(input_ch)

        in_chanel = input_ch + 6
        # Encoder[b,n,c]
        self.sa1 = SetAbstraction(1024, 0.1, 32, in_chanel, [64, 64, 128]) #3+64
        self.sa2 = SetAbstraction(256, 0.2, 32, 131, [128, 128, 256]) #128
        self.sa3 = SetAbstraction(64, 0.4, 32, 259, [256, 256, 512]) #256*2

        # 1st layer: input = 3(xyz) + 3(RGB) + 64(pos_encoding) = 70
        #self.sa1 = MultiScaleSetAbstraction(1024, [0.1, 0.2],[16, 32], 6+64, [64, 64, 128])
        # 2nd layer: input = 128*2 (Multi-scale connection)
        #self.sa2 = MultiScaleSetAbstraction(256,[0.2, 0.4],[16, 32], 128*2+3,[128, 128, 256])
        #self.sa3 = MultiScaleSetAbstraction(64,[0.4, 0.8],[16, 32], 256*2+3,[256, 256, 512])
        #  npoint=1024,radius_list=[0.1, 0.2], nsample_list=[16, 32], in_channel=6+64, mlp=[64, 64, 128]

        # attention module
        self.attention1 = EnhancedAttentionModule(256) #128 * 2
        self.attention2 = EnhancedAttentionModule(256 * 2)
        self.attention3 = EnhancedAttentionModule(512 * 2)

        # Geometric feature extraction
        self.geometric1 = GeometricFeatureExtraction(128 * 2)
        self.geometric2 = GeometricFeatureExtraction(256 * 2)
        self.geometric3 = GeometricFeatureExtraction(512 * 2)

        # Boundary aware modules
        self.boundary1 = BoundaryAwareModule(128)  # 256 channels * 2
        self.boundary2 = BoundaryAwareModule(256)  # 512 channels * 2
        self.boundary3 = BoundaryAwareModule(512)  # 1024 channels * 2

        # Decoder
        self.fp3 = FeaturePropagation(768, [256, 256])  # 512*2 + 256*2 1536
        self.fp2 = FeaturePropagation(384, [256, 128])  # 256 + 128*2 512
        self.fp1 = FeaturePropagation(128, [128, 128, 128]) # 128 + 64 128

        # Final layers
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, colors):
        """
        xyz: [B, N, 3]
        colors: [B, N, 3] (RGB)
        """

        # Add positional encoding
        pos_enc = self.pos_encoding(xyz) # [B, 64, N]
        # Change the order of dimensions
        colors = colors.transpose(1, 2)  # [B, 3, N]

        points = torch.cat([colors, pos_enc], dim=1)  # Merge Features: [B, 67, N]

        # Encoder with multi-scale feature extraction
        l1_xyz, l1_points = self.sa1(xyz, points)   #[B,70,N] -> [B, 128, N]
        # l1_points shape: [B, 256, N] (128*2 channels)
        #l1_points = self.attention1(l1_points)
        #l1_points = self.geometric1(l1_points, l1_xyz)
        #l1_points = self.boundary1(l1_points, l1_xyz)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # l2_points shape: [B, 512, N] (256*2 channels)
        #l2_points = self.attention2(l2_points)
        #l2_points = self.geometric2(l2_points, l2_xyz)
        #l2_points = self.boundary2(l2_points, l2_xyz)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # l3_points shape: [B, 1024, N] (512*2 channels)
        #l3_points = self.attention3(l3_points)
        #l3_points = self.geometric3(l3_points, l3_xyz)
        #l3_points = self.boundary3(l3_points, l3_xyz)

        # Decoder
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        feat = self.drop1(feat)
        out = self.conv2(feat)

        return out