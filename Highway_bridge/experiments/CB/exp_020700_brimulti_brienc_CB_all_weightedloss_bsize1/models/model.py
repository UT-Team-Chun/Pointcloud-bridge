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


import torch
import torch.nn as nn
import torch.nn.functional as F

class BridgeStructureLoss(nn.Module):
    def __init__(self, num_classes=5, alpha=20.0, rel_margin=0.2, class_weights=None, lambda_iou=0.5):
        """
        参数说明:
         - num_classes: 类别数，默认5。
         - alpha: 对于违反物理位置信息约束的惩罚系数。
         - rel_margin: 期望的最小相对位置差距。
         - class_weights: 可以传入自定义的类别权重。
         - lambda_iou: IoU损失项的权重，用于补充交叉熵损失，使miou更直接受损失优化。
        """
        super().__init__()
        self.alpha = alpha
        self.rel_margin = rel_margin
        self.lambda_iou = lambda_iou

        # 定义部件的层级关系
        self.hierarchy = {
            1: {'name': 'abutment', 'below': [2, 3], 'require': []},
            2: {'name': 'girder', 'above': [1], 'below': [3], 'require': [1]},
            3: {'name': 'deck', 'above': [2], 'below': [4], 'require': [2]},
            4: {'name': 'parapet', 'above': [2,3], 'require': []},  # 放宽类4的约束条件
            0: {'name': 'other'}
        }
        default_weights = torch.tensor([1.5, 1.0, 1.2, 1.5, 1.0])
        self.base_weights = default_weights if class_weights is None else class_weights
        self.register_buffer('base_weights_buffer', self.base_weights)

    def _get_relative_position(self, points, mask):
        """
        计算给定mask中所有点的归一化Z轴平均值
        参数:
         - points: 原始点云数据，形状为 (B, N, 3)
         - mask: 布尔型mask，形状为 (B, N)，表示当前类别的点
        """
        # 确保mask为float类型
        mask = mask.float()
        masked_points = points * mask.unsqueeze(-1)
        min_vals = masked_points.amin(dim=1, keepdim=True)
        max_vals = masked_points.amax(dim=1, keepdim=True)
        range_vals = max_vals - min_vals + 1e-7
        rel_pos = (masked_points - min_vals) / range_vals
        # 针对Z轴进行平均
        z_mean = (rel_pos[..., 2] * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return z_mean

    def compute_iou_loss(self, outputs, labels, epsilon=1e-6):
        """
        计算 IoU 损失项，期望输出经过 softmax。
        参数:
         - outputs: model logits，形状 (B, N, C)
         - labels: 真实标签，形状 (B, N)
        """
        # 将 logits 转 softmax 得到概率分布
        probs = F.softmax(outputs, dim=-1)  # shape: (B, N, C)
        # 转为 one hot格式
        labels_one_hot = F.one_hot(labels, num_classes=probs.shape[-1]).float()  # shape: (B, N, C)
        # 转置以便在类别维度上计算 (B, C, N)
        probs = probs.transpose(1, 2)
        labels_one_hot = labels_one_hot.transpose(1, 2)
        intersection = (probs * labels_one_hot).sum(dim=2)
        union = (probs + labels_one_hot - probs * labels_one_hot).sum(dim=2)
        iou = (intersection + epsilon) / (union + epsilon)
        # IoU越大越好，因此损失为1 - IoU
        loss = 1 - iou.mean()
        return loss

    def forward(self, outputs, labels, points):
        """
        参数:
         - outputs: 网络输出logits，原始形状 (B, C, N)
         - labels: 真实标签，形状 (B, N)
         - points: 点云数据，形状 (B, N, 3)
        """
        # 按原先方式，将outputs转为 (B, N, C)
        outputs = outputs.transpose(1, 2)
        B, N, C = outputs.shape
        device = outputs.device
        preds = torch.argmax(outputs, dim=-1)  # 预测类别，形状: (B, N)

        # 初始类别权重，按批次复制
        weights = self.base_weights_buffer.repeat(B, 1).to(device)

        # 针对部件类别构建存在性mask（除0号类）
        exist_mask = {
            cid: (labels == cid).float().sum(dim=1) > 0
            for cid in [1, 2, 3, 4]
        }

        # 计算每个部件类别在预测中的归一化z均值
        rel_pos = {}
        for cid in [1, 2, 3, 4]:
            mask = (preds == cid)
            # 如果有对应点，则调用相对位置计算函数
            rel_pos[cid] = self._get_relative_position(points, mask) if mask.any() else torch.zeros(B, device=device)

        # 根据层级关系调整类别权重
        for cid in [2, 3, 4]:
            info = self.hierarchy.get(cid, {})
            # 仅当必要的上层部件存在时，才进行约束
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

        # 对于其他类别（0类）的调整，根据在预测中出现的比例
        other_pred = (preds == 0).float().mean(dim=1)
        weights[:, 0] += self.alpha * (1 - other_pred)

        # 根据真实标签统计类别频次，动态调整类别权重
        class_dist = torch.bincount(labels.view(-1), minlength=C).float().clamp(min=1)
        dynamic_class_weights = (1 / class_dist.sqrt()).to(device)
        dynamic_class_weights[1] *= 2.0  # 增强类1权重
        dynamic_class_weights[4] *= 2.0  # 增强类4权重

        # 计算交叉熵损失
        ce_loss = F.cross_entropy(
            outputs.reshape(-1, C),
            labels.reshape(-1),
            weight=weights.mean(dim=0) * dynamic_class_weights,
            label_smoothing=0.2
        )

        # 计算 IoU 损失，作为辅助损失项
        iou_loss_val = self.compute_iou_loss(outputs, labels)

        total_loss = ce_loss + self.lambda_iou * iou_loss_val

        return total_loss



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
    #loss_fun = nn.CrossEntropyLoss()
    class_weights = torch.tensor([0.5, 2.2, 1.0, 1.2, 3.0])
    criterion = BridgeStructureLoss(
        num_classes=num_classes,
        base_alpha=10,
        strict_alpha=30,
        margin=0.05,
        density_threshold=0.1,
        class_weights_raw=class_weights,
        label_smoothing=0.1
    )

    main_model = main_model.to(device)
    main_model.eval()

    # 1. test forward pass
    print("=" * 50)
    print("モデルのテストを始まり")
    print(f"Inputs： xyz shape: {xyz.shape}")
    print(f"Output： features shape: {features.shape}")

    for epoch in range(1, 3):
        try:
            xyz = xyz.to(device) # [B, N, 3]
            features = features.to(device) # [B, N, 3]
            outputs = main_model(xyz, features) # [B, N, num_classes]
            print(f"main model xyz shape: {xyz.shape}")
            print(f"main model labels shape: {labels.shape}")
            print(f"main model output shape: {outputs.shape}")
            print("モデルの前伝播が成功した!")
            loss = criterion(outputs, labels, xyz)
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
