# models/enhanced_pointnet2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

from .pointnet2_utils import FeaturePropagation, SetAbstraction

class PointNet2(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        # Input channels for original features (e.g., RGB)
        # Assuming the input 'points' are RGB, so 3 channels.
        # If 'points' include other features, adjust this value.
        original_feature_channels = 3 

        # Encoder
        # The input_channel for sa1 should account for original features + xyz
        self.sa1 = SetAbstraction(1024, 0.1, 32, original_feature_channels + 3, [64, 64, 128]) # Input: xyz(3) + features(3) = 6
        self.sa2 = SetAbstraction(256, 0.2, 32, 128 + 3, [128, 128, 256]) # Input: l1_points(128) + l1_xyz(3) = 131
        self.sa3 = SetAbstraction(64, 0.4, 32, 256 + 3, [256, 256, 512]) # Input: l2_points(256) + l2_xyz(3) = 259

        # Decoder
        self.fp3 = FeaturePropagation(768, [256, 256]) # Input: l3_points(512) + l2_points(256) = 768
        self.fp2 = FeaturePropagation(384, [256, 128]) # Input: fp3_out(256) + l1_points(128) = 384
        # Modify fp1 in_channel: fp2_out(128) + original_features(3) = 131
        self.fp1 = FeaturePropagation(128 + original_feature_channels, [128, 128, 128]) # Input: fp2_out(128) + original_features(3) = 131

        # Final layers
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, points):
        """
        xyz: [B, N, 3]
        points: [B, N, 3] (RGB) or other features
        """

        # Change the order of dimensions for input features
        points_transposed = points.transpose(1, 2)  # [B, C_in, N]

        # Encoder with multi-scale feature extraction
        # Pass original features (transposed) to sa1
        l1_xyz, l1_points = self.sa1(xyz, points_transposed)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Decoder
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # Pass original features (transposed) to fp1 as points1
        l0_points = self.fp1(xyz, l1_xyz, points_transposed, l1_points)

        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        feat = self.drop1(feat)
        out = self.conv2(feat)

        return out