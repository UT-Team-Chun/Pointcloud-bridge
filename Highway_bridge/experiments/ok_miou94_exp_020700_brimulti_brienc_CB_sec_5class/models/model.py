# models/enhanced_pointnet2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .attention_modules import GeometricFeatureExtraction, EnhancedPositionalEncoding, BridgeStructureEncoding, \
    ColorFeatureExtraction, CompositeFeatureFusion
from .pointnet2_utils import FeaturePropagation, SetAbstraction, MultiScaleSetAbstraction, EnhancedFeaturePropagation


# from knn_cuda import KNN


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
        # 1st layer: input = 3(xyz) + 3(RGB) + 64(pos_encoding) = 70
        self.sa1 = MultiScaleSetAbstraction(1024, [0.1, 0.2],[16, 32], in_chanel, [64, 64, 128])
        # 2nd layer: input = 128*2 (Multi-scale connection)
        self.sa2 = MultiScaleSetAbstraction(512,[0.2, 0.4],[16, 32], 259,[128, 128, 256])
        self.sa3 = MultiScaleSetAbstraction(128,[0.4, 0.8],[16, 32], 515,[256, 256, 512])
        #self.sa4 = MultiScaleSetAbstraction(128,[0.8, 1.6],[16, 32], 1027,[512, 512, 1024])

        # attention module
        #self.attention1 = EnhancedAttentionModule(128*2) #128 * 2
        #self.attention2 = EnhancedAttentionModule(256*2)
        #self.attention3 = EnhancedAttentionModule(512*2)

        # Geometric feature extraction
        self.geometric1 = GeometricFeatureExtraction(128 * 2)
        self.geometric2 = GeometricFeatureExtraction(256 * 2)
        self.geometric3 = GeometricFeatureExtraction(512 * 2)

        # Boundary aware modules
        #self.boundary1 = BoundaryAwareModule(128*2)  # 256 channels * 2
        #self.boundary2 = BoundaryAwareModule(256*2)  # 512 channels * 2
        #self.boundary3 = BoundaryAwareModule(512*2)  # 1024 channels * 2

        # Decoder
        #self.fp4 = FeaturePropagation(3072, [1024, 512]) # multi: 1024*2 + 512*2 ,3072; 512 + 512*2 ,1536
        self.fp3 = EnhancedFeaturePropagation(1536, [1024, 256])  # multi: 512*2 + 256*2 ,1536
        self.fp2 = EnhancedFeaturePropagation(512, [256, 256])  # multi:  256*2 ,512
        self.fp1 = EnhancedFeaturePropagation(256, [256, 128]) # multi:  128*2 ,256

        self.fusion = MultiScaleFeatureFusion(
            in_channels_list=[256, 256, 128],  # fp3, fp2, fp1的输出通道
            out_channels=128
        )
        # 添加多尺度特征聚合 Final layers
        self.final_fusion = nn.Sequential(
            nn.Conv1d(384, 128, 1), #128+256+256
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1)
        )
        #
        # # Final layers
        # self.conv1 = nn.Conv1d(128, 128, 1)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.drop1 = nn.Dropout(0.5)
        # self.conv2 = nn.Conv1d(128, num_classes, 1)

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
        #l1_points = self.attention1(l1_points)
        l1_features = self.geometric1(l1_features, l1_xyz)
        #l1_points = self.boundary1(l1_points, l1_xyz)

        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        # l2_points shape: [B, 512, N] (256*2 channels)
        #l2_points = self.attention2(l2_points)
        l2_features = self.geometric2(l2_features, l2_xyz)
        #l2_points = self.boundary2(l2_points, l2_xyz)

        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)
        # l3_points shape: [B, 1024, N] (512*2 channels)
        #l3_points = self.attention3(l3_points)
        l3_features = self.geometric3(l3_features, l3_xyz)
        #l3_points = self.boundary3(l3_points, l3_xyz)

        #l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Decoder
        #l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_features = self.fp3(l2_xyz, l3_xyz, l2_features, l3_features)
        l1_features = self.fp2(l1_xyz, l2_xyz, l1_features, l2_features)
        l0_features = self.fp1(xyz, l1_xyz, None, l1_features)

        fused_features=self.fusion([l2_features, l1_features, l0_features])

        # # 多尺度特征聚合
        # l2_upsampled = F.interpolate(l2_features, size=l0_features.shape[2])
        # l1_upsampled = F.interpolate(l1_features, size=l0_features.shape[2])
        #
        # multi_scale_features = torch.cat([
        #     l0_features,
        #     l1_upsampled,
        #     l2_upsampled
        # ], dim=1)

        # 最终分类
        x = self.final_fusion(fused_features)

        # # FC layers
        # feat = F.relu(self.bn1(self.conv1(l0_features)))
        # feat = self.drop1(feat)
        # x = self.conv2(feat)


        return x


class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ))

    def forward(self, features_list):
        out_features = []
        l0_features = features_list[2]
        for feat, conv in zip(features_list, self.convs):
            feat = F.interpolate(feat, size=l0_features.shape[2])
            out_features.append(conv(feat))

        return torch.cat(out_features, dim=1)

class BridgeStructureLoss(nn.Module):
    def __init__(self, num_classes=5, alpha=20.0, rel_margin=0.2, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.rel_margin = rel_margin

        # 定义层级关系
        self.hierarchy = {
            1: {'name': 'abutment', 'below': [2, 3], 'require': []},
            2: {'name': 'girder', 'above': [1], 'below': [3], 'require': [1]},
            3: {'name': 'deck', 'above': [2], 'below': [4], 'require': [2]},
            4: {'name': 'parapet', 'above': [3], 'require': []},  # 放宽类4的约束条件
            0: {'name': 'other'}
        }

        # 默认类别权重
        default_weights = torch.tensor([1.5, 1.0, 1.2, 1.5, 1.0])
        self.base_weights = default_weights if class_weights is None else class_weights
        self.register_buffer('base_weights_buffer', self.base_weights)

    def _get_relative_position(self, points, mask):
        masked_points = points * mask.unsqueeze(-1)
        min_vals = masked_points.amin(dim=1, keepdim=True)
        max_vals = masked_points.amax(dim=1, keepdim=True)
        range_vals = max_vals - min_vals + 1e-7
        rel_pos = (masked_points - min_vals) / range_vals
        z_mean = (rel_pos[..., 2] * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return z_mean

    def forward(self, outputs, labels, points):
        outputs = outputs.transpose(1, 2)
        B, N = labels.shape
        device = outputs.device
        preds = torch.argmax(outputs, dim=-1)

        weights = self.base_weights_buffer.repeat(B, 1).to(device)

        # ============ 存在性检查 ============
        exist_mask = {
            cid: (labels == cid).float().sum(dim=1) > 0
            for cid in [1, 2, 3, 4]
        }

        rel_pos = {}
        for cid in [1, 2, 3, 4]:
            mask = preds == cid
            rel_pos[cid] = self._get_relative_position(points, mask) if mask.any() else torch.zeros(B, device=device)

        # ============ 修复层级约束逻辑 ============
        for cid in [2, 3, 4]:
            info = self.hierarchy.get(cid, {})

            # 使用 any() 进行存在性判断
            required = all([exist_mask[rid].any().item() for rid in info.get('require', [])])
            if not required:
                continue

            if 'above' in info:
                for lower_cid in info['above']:
                    if not exist_mask.get(lower_cid, torch.tensor(False)).any():
                        continue

                    pos_diff = rel_pos[cid] - rel_pos[lower_cid]
                    violation = F.relu(-pos_diff + self.rel_margin)
                    weights[:, cid] += self.alpha * violation
                    weights[:, lower_cid] += self.alpha * violation * 0.5

            if 'below' in info:
                for upper_cid in info['below']:
                    if not exist_mask.get(upper_cid, torch.tensor(False)).any():
                        continue

                    pos_diff = rel_pos[upper_cid] - rel_pos[cid]
                    violation = F.relu(-pos_diff + self.rel_margin)
                    weights[:, cid] += self.alpha * violation
                    weights[:, upper_cid] += self.alpha * violation * 0.3

        # ============ 类别权重动态调整 ============
        other_pred = (preds == 0).float().mean(dim=1)
        weights[:, 0] += self.alpha * (1 - other_pred)

        class_dist = torch.bincount(labels.view(-1), minlength=5).float().clamp(min=1)
        class_weights = (1 / class_dist.sqrt()).to(device)
        class_weights[1] *= 2.0  # 增加类1的权重
        class_weights[4] *= 2.0  # 增加类4的权重

        return F.cross_entropy(
            outputs.reshape(-1, 5),
            labels.reshape(-1),
            weight=weights.mean(dim=0) * class_weights,
            label_smoothing=0.2
        )

# 创建一个简单的数据集类
class RandomPointCloudDataset(Dataset):
    def __init__(self, num_samples=1000, num_points=1024):
        self.num_samples = num_samples
        self.num_points = num_points

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成随机点云数据
        xyz = torch.randn(self.num_points, 3)  # [N, 3]
        colors = torch.rand(self.num_points, 3)  # [N, 3]

        # 确保数据类型和形状正确
        xyz = xyz.float()
        colors = colors.float()

        return {
            'xyz': xyz,  # [N, 3]
            'colors': colors  # [N, 3]
        }


if __name__ == "__main__":
    # 设置随机种子保证可复现性
    torch.manual_seed(42)

    # 创建测试数据
    batch_size = 2
    num_points = 1024

    # 创建数据集和数据加载器
    dataset = RandomPointCloudDataset(num_samples=100, num_points=num_points)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # 创建随机输入数据
    xyz = torch.randn(batch_size, num_points, 3)
    features = torch.randn(batch_size, num_points, 3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 创建预训练模型
    pretrain_model = PointCloudPretraining()
    pretrain_model = pretrain_model.to(device)

    # 2. 进行预训练 (设置较小的epoch数用于测试)
    pretrain_model = pretrain(pretrain_model, train_loader, epochs=2, device=device)

    # 3. 将预训练权重迁移到主模型
    main_model = EnhancedPointNet2() #EnhancedPointNet2
    # 复制编码器权重
    # main_model.pos_encoding.load_state_dict(pretrain_model.pos_encoding.state_dict())
    # main_model.bri_enc.load_state_dict(pretrain_model.bri_enc.state_dict())
    # main_model.color_encoder.load_state_dict(pretrain_model.color_encoder.state_dict())
    # main_model.feature_fusion.load_state_dict(pretrain_model.feature_fusion.state_dict())
    # main_model.sa1.load_state_dict(pretrain_model.sa1.state_dict())
    # main_model.sa2.load_state_dict(pretrain_model.sa2.state_dict())
    # main_model.sa3.load_state_dict(pretrain_model.sa3.state_dict())
    # main_model.geometric1.load_state_dict(pretrain_model.geometric1.state_dict())
    # main_model.geometric2.load_state_dict(pretrain_model.geometric2.state_dict())
    # main_model.geometric3.load_state_dict(pretrain_model.geometric3.state_dict())

    # 将主模型移到设备上
    pretrain_model = pretrain_model.to(device)
    pretrain_model.eval()
    main_model = main_model.to(device)
    main_model.eval()

    # 1. 基础功能测试
    print("=" * 50)
    print("基础功能测试")
    print(f"输入 xyz shape: {xyz.shape}")
    print(f"输入 features shape: {features.shape}")

    try:
        xyz = xyz.to(device)
        features = features.to(device)
        output = main_model(xyz, features)
        reconstructed_xyz, predicted_quaternion= pretrain_model(xyz, features)
        print(f"main model 输出 shape: {output.shape}")
        print(f"pretrain model 输出 shape: {reconstructed_xyz.shape}")
        print("模型前向传播测试通过!")
    except Exception as e:
        print(f"模型运行出错: {str(e)}")

    # 2. 模型信息统计
    print("\n" + "=" * 50)
    print("模型信息统计")

    total_params = sum(p.numel() for p in main_model.parameters())
    trainable_params = sum(p.numel() for p in main_model.parameters() if p.requires_grad)

    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 7. 测试不同batch size的内存占用
    print("\n" + "=" * 50)
    print("内存占用测试")

    batch_sizes = [4, 16]
    for bs in batch_sizes:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 清空GPU缓存
            xyz_test = torch.randn(bs, num_points, 3).to(device)
            features_test = torch.randn(bs, num_points, 3).to(device)

            torch.cuda.reset_peak_memory_stats()
            output = main_model(xyz_test, features_test)
            memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # 转换为MB
            print(f"Batch size {bs}: 峰值显存占用 {memory:.2f} MB")

    print("\n测试完成!")
