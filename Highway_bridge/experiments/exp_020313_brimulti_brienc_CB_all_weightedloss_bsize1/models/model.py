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


class BridgeStructureLoss(nn.Module):
    """
    优化思路：
      1. 用原始z轴统计数据衡量物理部件的位置关系；
      2. 对于需要严格满足的层级关系（例如 girder 应该位于pier之上、deck 在girder之上、parapet在deck之上），
         当平均z值差距低于预设margin时进行惩罚（严格与基础惩罚区别）；
      3. 加入孤立点抑制，避免背景错误干扰；
      4. 采用动态权重，结合类别样本数及层级惩罚修正。

    层级关系：
      - 类1 (piers): 要求在其上部出现 girder/deck/parapet 时，保持一定差距；
      - 类2 (girders): 必须在pier之上且deck之下；
      - 类3 (deck): 必须在girder之下且parapet之下；
      - 类4 (parapet): 应该在上部，允许有较宽松的容忍度。
    """

    def __init__(self, num_classes=5, base_alpha=15.0, strict_alpha=30.0, margin=0.05, density_threshold=0.1,
                 class_weights_raw=None, label_smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.base_alpha = base_alpha  # 基础加权系数
        self.strict_alpha = strict_alpha  # 严重违规的加权系数
        self.margin = margin  # 需要满足的最小高度差
        self.density_threshold = density_threshold
        self.label_smoothing = label_smoothing

        # 层级约束定义，注意：关系中的数字代表其他类别ID
        # strict_above: 本部件需要低于这些部件（即本部件平均z值要低于对方）
        # strict_below: 本部件需要高于这些部件（即本部件平均z值要高于对方）
        self.hierarchy = {
            1: {'strict_below': [2, 3], 'soft_below': [4]},  # pier: 一般要求其上方出现的girder和deck
            2: {'strict_above': [1], 'strict_below': [3]},  # girder: 在pier之上，在deck之下
            3: {'strict_above': [2], 'strict_below': [4]},  # deck: 在girder之上，在parapet之下
            4: {'strict_above': [1, 2, 3]},  # parapet: 通常应位于其他结构的上方
            0: {}
        }
        # 类别权重：若用户未指定则采用默认值
        if class_weights_raw is None:
            # 可根据训练集类别数量调整下面的数值
            class_weights_raw = torch.tensor([1.5, 1.0, 1.2, 1.5, 1.0])
        self.register_buffer('class_weights', class_weights_raw)

    def _get_z_stats(self, points, mask):
        """
        根据mask（B,N）计算每个样本的z最小值、最大值、均值
        注意：mask为float型，0或1值
        """
        # 仅对被mask到的点进行计算
        masked_z = points[..., 2] * mask
        valid_count = mask.sum(dim=1).clamp(min=1)
        z_min = masked_z.min(dim=1)[0]
        z_max = masked_z.max(dim=1)[0]
        z_mean = masked_z.sum(dim=1) / valid_count
        valid = valid_count > 0
        return z_min, z_max, z_mean, valid

    def forward(self, outputs, labels, points):
        # outputs形状 [B, C, N]，labels [B, N]，points [B, N, 3]
        B, C, N = outputs.shape
        device = outputs.device

        # 预测类别
        preds = torch.argmax(outputs, dim=1)  # [B, N]

        # 初始化类别权重（先拷贝基础类别权重，每个样本）
        weights = self.class_weights.repeat(B, 1).to(device)

        # 存在性检测：对于每个部件类型，若该部件在labels中密度低于阈值，则认为不存在
        exist_mask = {}
        for cid in [1, 2, 3, 4]:
            exist_mask[cid] = (labels == cid).float().mean(dim=1) > self.density_threshold

        # --- 层级约束 ---
        # 针对每个类别，根据预测结果计算z均值
        z_means = {}
        valid_flags = {}
        for cid in [1, 2, 3, 4]:
            mask = (preds == cid).float()
            _, _, z_mean, valid = self._get_z_stats(points, mask)
            z_means[cid] = z_mean
            valid_flags[cid] = valid

        # 针对每个部件，依次依据层级关系调整权重
        for cid in [1, 2, 3, 4]:
            # 判断当前类别是否存在（预测与GT均考虑）
            if not exist_mask[cid].any():
                continue

            # 对于严格约束关系（如strict_above 和 strict_below），计算高度差，并判断是否满足margin
            if 'strict_above' in self.hierarchy[cid]:
                for other_cid in self.hierarchy[cid]['strict_above']:
                    # 若其他类别在当前batch中也存在
                    if not exist_mask.get(other_cid, torch.tensor(False)).any():
                        continue
                    # 计算当前类别与other类别的高度差：要求 other 部件的z均值需高于当前部件（z_diff > margin）
                    diff = z_means[other_cid] - z_means[cid]
                    violation = F.relu(self.margin - diff)  # 当实际差值diff不足margin，则产生违例
                    # 使用严格惩罚系数
                    alpha = self.strict_alpha
                    # 更新当前样本中两个类别的权重（仅对满足条件的batch样本更新）
                    valid = valid_flags[cid] & valid_flags[other_cid]
                    if valid.any():
                        weights[valid, cid] += alpha * violation[valid]
                        weights[valid, other_cid] += 0.5 * alpha * violation[valid]

            if 'strict_below' in self.hierarchy[cid]:
                for other_cid in self.hierarchy[cid]['strict_below']:
                    if not exist_mask.get(other_cid, torch.tensor(False)).any():
                        continue
                    # 此处要求当前部件的z均值应高于 other 部件（z_diff > margin）
                    diff = z_means[cid] - z_means[other_cid]
                    violation = F.relu(self.margin - diff)
                    alpha = self.strict_alpha
                    valid = valid_flags[cid] & valid_flags[other_cid]
                    if valid.any():
                        weights[valid, cid] += alpha * violation[valid]
                        weights[valid, other_cid] += 0.5 * alpha * violation[valid]

            # 对于软约束（例如pier对parapet的soft_below），惩罚系数可稍小，容忍度可适当放宽
            if 'soft_below' in self.hierarchy[cid]:
                for other_cid in self.hierarchy[cid]['soft_below']:
                    if not exist_mask.get(other_cid, torch.tensor(False)).any():
                        continue
                    diff = z_means[cid] - z_means[other_cid]
                    violation = F.relu(self.margin * 0.5 - diff)
                    alpha = self.base_alpha  # 使用基础惩罚
                    valid = valid_flags[cid] & valid_flags[other_cid]
                    if valid.any():
                        weights[valid, cid] += alpha * violation[valid]
                        weights[valid, other_cid] += 0.5 * alpha * violation[valid]

        # 背景误判抑制：计算每个样本中非背景点（类别不为0）的全局比例
        global_non_bg_density = (preds != 0).float().mean(dim=1)  # shape: [B]
        for b in range(B):
            if global_non_bg_density[b] < 0.1:  # 如果该样本中非背景点比例较低
                weights[b, 0] += self.base_alpha * 0.8  # 增加背景权重
                # 同时对其它类别加大权重，防止背景“抢占”
                weights[b, 1:] += self.base_alpha * 1.2

        # 动态归一化权重（防止权重太大或太小）
        weights = torch.clamp(weights, min=0.5, max=5.0)

        # 为防止类别不平衡，再结合一次统计，计算各类别的频次作为进一步校正
        counts = torch.bincount(labels.view(-1), minlength=self.num_classes).float().clamp(min=1)
        dynamic_class_weights = (1 / counts.sqrt()).to(device)
        # 根据经验可对关键部件（例如pier和parapet）强化
        dynamic_class_weights[1] *= 2.0
        dynamic_class_weights[4] *= 2.0

        # 最终进行交叉熵损失计算，使用label smoothing
        loss = F.cross_entropy(
            outputs,  # shape: [B, C, N]
            labels,  # shape: [B, N]
            weight=(weights.mean(dim=0) * dynamic_class_weights),
            label_smoothing=self.label_smoothing
        )
        return loss


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
