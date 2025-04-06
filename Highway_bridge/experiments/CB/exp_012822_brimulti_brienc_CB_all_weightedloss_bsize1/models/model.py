# models/enhanced_pointnet2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .attention_modules import GeometricFeatureExtraction, EnhancedPositionalEncoding, BridgeStructureEncoding, \
    ColorFeatureExtraction, \
    CompositeFeatureFusion
from .pointnet2_utils import FeaturePropagation, SetAbstraction, MultiScaleSetAbstraction, PointTransformerFusion, \
    EnhancedFeaturePropagation


class PointNet2(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        # Encoder
        self.sa1 =SetAbstraction(1024, 0.1, 32, 6, [64, 64, 128]) #
        self.sa2 = SetAbstraction(256, 0.2, 32,131 , [128, 128, 256]) #128+3=131
        self.sa3 = SetAbstraction(64, 0.4, 32, 259, [256, 256, 512]) #

        # Decoder
        self.fp3 = FeaturePropagation(768, [256, 256]) # 512+256=768
        self.fp2 = FeaturePropagation(384, [256, 128]) # 256+128=384
        self.fp1 = FeaturePropagation(128, [128, 128, 128]) # 128+3=131

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

        # Change the order of dimensions
        points = points.transpose(1, 2)  # [B, 3, N]

        # Encoder with multi-scale feature extraction
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Decoder
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        feat = self.drop1(feat)
        out = self.conv2(feat)

        return out


class EnhancedPointNet2(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        input_ch=29
        self.pos_encoding = EnhancedPositionalEncoding(input_ch,4,64,)
        self.bri_enc = BridgeStructureEncoding(input_ch, 32, 4)

        # 颜色特征处理模块
        self.color_encoder = ColorFeatureExtraction(3, 32)
        self.feature_fusion = CompositeFeatureFusion(input_ch, 32)

        in_chanel = input_ch + 3 # 3(xyz) + 3(RGB)

        # Encoder

        # 3layer ty
        self.sa1 = MultiScaleSetAbstraction(1024, [0.1, 0.2],[16, 32], in_chanel, [64, 64, 128])
        self.sa2 = MultiScaleSetAbstraction(512,[0.2, 0.4],[16, 32], 259,[128, 128, 256])
        self.sa3 = MultiScaleSetAbstraction(128,[0.4, 0.8],[16, 32], 515,[256, 256, 512])
        self.geometric1 = GeometricFeatureExtraction(128 * 2)
        self.geometric2 = GeometricFeatureExtraction(256 * 2)
        self.geometric3 = GeometricFeatureExtraction(512 * 2)
        # Decoder
        self.fp3 = EnhancedFeaturePropagation(1536, [1024, 256])  # multi: 512*2 + 256*2 ,1536
        self.fp2 = EnhancedFeaturePropagation(512, [256, 256])  # multi:  256*2 ,512
        self.fp1 = EnhancedFeaturePropagation(256, [256, 128]) # multi:  128*2 ,256

        # 4 layer
        # # 1st layer: input = 3(xyz) + 3(RGB) + 64(pos_encoding) = 70
        # self.sa1 = MultiScaleSetAbstraction(2048, [0.1, 0.2],[16, 32], in_chanel, [64, 64, 128])
        # # 2nd layer: input = 128*2 (Multi-scale connection)
        # self.sa2 = MultiScaleSetAbstraction(1024,[0.2, 0.4],[16, 32], 259,[128, 128, 256])
        # self.sa3 = MultiScaleSetAbstraction(512,[0.4, 0.8],[16, 32], 515,[256, 256, 512])
        # self.sa4 = MultiScaleSetAbstraction(128,[0.8, 1.6],[16, 32], 1027,[512, 512, 1024])
        #
        # # Geometric feature extraction
        # self.geometric1 = GeometricFeatureExtraction(128 * 2)
        # self.geometric2 = GeometricFeatureExtraction(256 * 2)
        # self.geometric3 = GeometricFeatureExtraction(512 * 2)
        # self.geometric4 = GeometricFeatureExtraction(1024 * 2)
        #
        # # Decoder
        # self.fp4 = EnhancedFeaturePropagation(3072, [1024, 512]) # multi: 1024*2 + 512*2 ,3072; 512 + 512*2 ,1536
        # self.fp3 = EnhancedFeaturePropagation(1024, [512, 256])  # multi: 512*2 + 256*2 ,1536
        # self.fp2 = EnhancedFeaturePropagation(512, [256, 128])  # multi:  256*2 ,512
        # self.fp1 = EnhancedFeaturePropagation(128, [128, 128]) # multi:  128*2 ,256

        # Transformer fusion
        self.feat_tf = PointTransformerFusion(
            feature_dims=[64, 128, 256, 512],  # 根据实际特征维度调整
            num_heads=4,
            unified_dim=128,
            output_dim=128,
            num_layers=2
        )

        # multi Final layers
        self.final_fusion = nn.Sequential(
            nn.Conv1d(640, 258, 1), #128+256+256; #64+128+256+512=960
            nn.BatchNorm1d(258),
            nn.ReLU(),
            nn.Conv1d(258, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1)
        )
        #
        # Final layers
        self.seg_head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, xyz, colors):

        """
        xyz: [B, N, 3]
        points: [B, N, 32] (RGB)
        """

        # Add positional encoding
        pos_enc = self.bri_enc(xyz) # [B, 64, N]
        # Change the order of dimensions
        colors = colors.transpose(1, 2)  # [B, 3, N]
        color_features = self.color_encoder(colors, xyz)  # [B, 12, N]
        fused_features = self.feature_fusion(pos_enc, color_features)  # [B, input_ch, N]
        #features = torch.cat([pos_enc, colors], dim=1)  # Merge Features: [B, 67, N]

        # Encoder with multi-scale feature extraction
        l1_xyz, l1_features= self.sa1(xyz, fused_features)   #[B,70,N] -> [B, 128, N]
        # l1_points shape: [B, 256, N] (128*2 channels)
        l1_features = self.geometric1(l1_features, l1_xyz)
        #l1_features = self.attention1(l1_features)
        #l1_points = self.boundary1(l1_points, l1_xyz)

        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        # l2_points shape: [B, 512, N] (256*2 channels)
        l2_features = self.geometric2(l2_features, l2_xyz)
        #l2_features = self.attention2(l2_features)
        #l2_points = self.boundary2(l2_points, l2_xyz)

        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)
        # l3_points shape: [B, 1024, N] (512*2 channels)
        l3_features = self.geometric3(l3_features, l3_xyz)
        #l3_features = self.attention3(l3_features)
        #l3_points = self.boundary3(l3_points, l3_xyz)

        #l4_xyz, l4_features = self.sa4(l3_xyz, l3_features)
        #l4_features = self.geometric4(l4_features, l4_xyz)


        # Decoder
        #l3_features = self.fp4(l3_xyz, l4_xyz, l3_features, l4_features)
        l2_features = self.fp3(l2_xyz, l3_xyz, l2_features, l3_features)
        l1_features = self.fp2(l1_xyz, l2_xyz, l1_features, l2_features)
        l0_features = self.fp1(xyz, l1_xyz, None, l1_features)


        # transformer fusion
        #fused_features = self.feat_tf(l0_features, l1_features, l2_features, l3_features)

        # # multi feature fusion
        #l3_upsampled = F.interpolate(l3_features, size=l0_features.shape[2])
        l2_upsampled = F.interpolate(l2_features, size=l0_features.shape[2])
        l1_upsampled = F.interpolate(l1_features, size=l0_features.shape[2])

        # Weighted sum
        multi_scale_features = torch.cat([
            l0_features,
            l1_upsampled,
            l2_upsampled
        ], dim=1)

        # final layers
        x = self.final_fusion(multi_scale_features)


        return x


# create a random point cloud dataset
class RandomPointCloudDataset(Dataset):
    def __init__(self, num_samples=1000, num_points=1024):
        self.num_samples = num_samples
        self.num_points = num_points

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        xyz = torch.randn(self.num_points, 3)  # [N, 3]
        colors = torch.rand(self.num_points, 3)  # [N, 3]

        xyz = xyz.float()
        colors = colors.float()

        return {
            'xyz': xyz,  # [N, 3]
            'colors': colors  # [N, 3]
        }


if __name__ == "__main__":

    torch.manual_seed(42)

    # create a random point cloud dataset
    batch_size = 2
    num_points = 1024

    # create a random point cloud dataset
    dataset = RandomPointCloudDataset(num_samples=100, num_points=num_points)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # create a random batch of point clouds
    xyz = torch.randn(batch_size, num_points, 3)
    features = torch.randn(batch_size, num_points, 3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    main_model = EnhancedPointNet2() #EnhancedPointNet2

    main_model = main_model.to(device)
    main_model.eval()

    # 1. test forward pass
    print("=" * 50)
    print("モデルのテストを始まり")
    print(f"Inputs： xyz shape: {xyz.shape}")
    print(f"Output： features shape: {features.shape}")

    try:
        xyz = xyz.to(device)
        features = features.to(device)
        output = main_model(xyz, features)
        print(f"main model input shape: {output.shape}")
        print("モデルの前伝播が成功した!")
    except Exception as e:
        print(f"モデルの実行がエラーがある: {str(e)}")

    # 2. model summary
    print("\n" + "=" * 50)
    print("モデルの情報")

    total_params = sum(p.numel() for p in main_model.parameters())
    trainable_params = sum(p.numel() for p in main_model.parameters() if p.requires_grad)

    print(f"Number of total parameters: {total_params:,}")
    print(f"Number of trainable parameters: {trainable_params:,}")

    # 7. test the memory usage of different batch sizes
    print("\n" + "=" * 50)
    print("Test the memory usage of different batch sizes")

    batch_sizes = [4, 16]
    for bs in batch_sizes:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            xyz_test = torch.randn(bs, num_points, 3).to(device)
            features_test = torch.randn(bs, num_points, 3).to(device)

            torch.cuda.reset_peak_memory_stats()
            output = main_model(xyz_test, features_test)
            memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"Batch size {bs}: max {memory:.2f} MB")

    print("\n Test complete!")
