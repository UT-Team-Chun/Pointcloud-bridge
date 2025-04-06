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


class BridgeStructureLoss(nn.Module):
    def __init__(self, num_classes=5, alpha=100.0, span_ratio=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha  # 降低惩罚强度
        self.span_ratio = span_ratio
        self.num_classes = num_classes

        # 初始化类别权重（包含other类）
        if class_weights is not None:
            self.base_weights = class_weights.clone().detach()
        else:
            self.base_weights = torch.ones(num_classes)

        # 注册为PyTorch buffer
        self.register_buffer('base_weights_buffer', self.base_weights)

        # 桥梁部件定义（包含other类处理）
        self.class_info = {
            0: {'name': 'other', 'penalty': 0.1},  # 降低other类的惩罚
            1: {'name': 'abutment', 'z_relations': {'below': [2, 3]}, 'penalty': 1.0},
            2: {'name': 'girder', 'xy_constraint': True, 'penalty': 1.0},
            3: {'name': 'deck', 'z_relations': {'above': [2], 'below': [4]}, 'penalty': 1.0},
            4: {'name': 'parapet', 'xy_constraint': True, 'penalty': 1.0}
        }

    def compute_span(self, points, mask):
        """向量化计算空间跨度的实用函数"""
        # points: [B, N, 3], mask: [B, N]
        masked_points = points * mask.unsqueeze(-1)  # [B, N, 3]

        # 计算有效点范围（使用大值/小值填充无效区域）
        large_num = 1e7
        x_valid = masked_points[..., 0] + (~mask) * large_num
        x_max, _ = torch.max(-x_valid, dim=1)  # 取反后max得到最小实际值
        x_min, _ = torch.min(x_valid, dim=1)
        x_span = (-x_max) - x_min

        y_valid = masked_points[..., 1] + (~mask) * large_num
        y_max, _ = torch.max(-y_valid, dim=1)
        y_min, _ = torch.min(y_valid, dim=1)
        y_span = (-y_max) - y_min

        return torch.stack([x_span, y_span], dim=1)  # [B, 2]

    def forward(self, outputs, labels, points):
        outputs = outputs.transpose(1, 2)  # [B, N, C]
        B, N = labels.shape
        device = outputs.device

        # 添加类别平衡权重
        class_counts = torch.bincount(labels.view(-1), minlength=self.num_classes).float()
        class_weights = self.base_weights_buffer.to(outputs.device)/ (class_counts + 1e-7)
        class_weights = class_weights / class_weights.sum() * self.num_classes

        # 初始化动态权重矩阵
        weights = class_weights.repeat(B, 1)  # [B, C]

        # 预测结果处理
        preds = torch.argmax(outputs, dim=-1)

        # 存在性检查（包含other类）
        gt_exist = {cid: (labels == cid).any(dim=1) for cid in self.class_info}
        pr_exist = {cid: (preds == cid).any(dim=1) for cid in self.class_info}

        # 误检惩罚调整（降低other类的惩罚）
        for cid, info in self.class_info.items():
            penalty = info.get('penalty', 1.0)
            mask = ~gt_exist[cid] & pr_exist[cid]
            weights[:, cid] += self.alpha * penalty * mask.float()

        # 几何约束优化
        for cid, info in self.class_info.items():
            if cid == 0 or not info.get('xy_constraint', False):
                continue

            # 使用相对误差代替绝对阈值
            with torch.no_grad():
                gt_mask = labels == cid
                gt_span = self.compute_span(points, gt_mask)
                pr_span = self.compute_span(points, preds == cid)

                span_ratio = (pr_span / (gt_span + 1e-7)).clamp(1 / self.span_ratio, self.span_ratio)
                violation = torch.abs(span_ratio - 1.0)  # [B, 2]

            weights[:, cid] += self.alpha * violation.mean(dim=1)

        # 层级关系约束优化（添加容错机制）
        gt_z = torch.stack([(points[..., 2] * (labels == cid)).sum(dim=1) /
                            (labels == cid).sum(dim=1).clamp(min=1e-7) for cid in self.class_info], dim=1)
        pr_z = torch.stack([(points[..., 2] * (preds == cid)).sum(dim=1) /
                            (preds == cid).sum(dim=1).clamp(min=1e-7) for cid in self.class_info], dim=1)

        for cid, info in self.class_info.items():
            if cid == 0 or 'z_relations' not in info:
                continue

            # 添加高度容差（允许10%的误差）
            if 'below' in info['z_relations']:
                for upper_cid in info['z_relations']['below']:
                    z_diff = (pr_z[:, cid] - pr_z[:, upper_cid]) / (gt_z[:, upper_cid] - gt_z[:, cid] + 1e-7)
                    violation = torch.sigmoid(z_diff * 5)  # 非线性惩罚
                    weights[:, cid] += self.alpha * 0.5 * violation
                    weights[:, upper_cid] += self.alpha * 0.5 * violation

            if 'above' in info['z_relations']:
                for lower_cid in info['z_relations']['above']:
                    z_diff = (pr_z[:, lower_cid] - pr_z[:, cid]) / (gt_z[:, cid] - gt_z[:, lower_cid] + 1e-7)
                    violation = torch.sigmoid(z_diff * 5) # 非线性渐变惩罚
                    weights[:, cid] += self.alpha * 0.5 * violation
                    weights[:, lower_cid] += self.alpha * 0.5 * violation

        # 添加other类正则化（防止全零预测）
        other_mask = (labels == 0).float().mean(dim=1)  # [B]
        weights[:, 0] += self.alpha * (1 - other_mask) * 0.1  # 动态调整

        return F.cross_entropy(
            outputs.reshape(-1, self.num_classes),
            labels.reshape(-1),
            weight=weights.mean(dim=0),
            label_smoothing=0.1  # 添加标签平滑
        )


if __name__ == "__main__":

    torch.manual_seed(42)

    # create a random point cloud dataset
    batch_size = 2
    num_points = 1024
    num_classes = 5

    # # create a random point cloud dataset
    # dataset = RandomPointCloudDataset(num_samples=100, num_points=num_points)
    # train_loader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=0
    # )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # create a random batch of point clouds
    xyz = torch.randn(batch_size, num_points, 3)
    features = torch.randn(batch_size, num_points, 3)
    labels = torch.randint(0, num_classes, (batch_size, num_points)).to(device)


    main_model = EnhancedPointNet2(num_classes) #EnhancedPointNet2
    loss_fun = nn.CrossEntropyLoss()
    class_weights = torch.tensor([0.5, 2.2, 1.0, 1.2, 3.0])
    criterion = BridgeStructureLoss(
        num_classes=5,
        alpha=100,        # 降低惩罚强度
        span_ratio=2.0,   # 放宽几何约束
        class_weights=class_weights  # 提升deck类的权重
        )

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
        outputs = main_model(xyz, features)
        print(f"main model labels shape: {labels.shape}")
        print(f"main model output shape: {outputs.shape}")
        print("モデルの前伝播が成功した!")
        loss = criterion(outputs, labels, points=xyz)
        #loss = loss_fun(outputs, labels)
        print(f"Loss shape: {loss}")
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
