# models/enhanced_pointnet2.py
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

        # 添加多尺度特征聚合 Final layers
        self.final_fusion = nn.Sequential(
            nn.Conv1d(640, 128, 1), #128+256+256
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

        # 多尺度特征聚合
        l2_upsampled = F.interpolate(l2_features, size=l0_features.shape[2])
        l1_upsampled = F.interpolate(l1_features, size=l0_features.shape[2])

        multi_scale_features = torch.cat([
            l0_features,
            l1_upsampled,
            l2_upsampled
        ], dim=1)

        # 最终分类
        x = self.final_fusion(multi_scale_features)

        # # FC layers
        # feat = F.relu(self.bn1(self.conv1(l0_features)))
        # feat = self.drop1(feat)
        # x = self.conv2(feat)


        return x


class PointCloudPretraining(nn.Module):
    def __init__(self):
        super().__init__()
        # 复制原始模型的编码器部分
        input_ch = 29
        self.pos_encoding = EnhancedPositionalEncoding(input_ch, 4, 64)
        self.bri_enc = BridgeStructureEncoding(input_ch, 32, 4)
        self.color_encoder = ColorFeatureExtraction(3, 32)
        self.feature_fusion = CompositeFeatureFusion(input_ch, 32)

        in_channel = input_ch + 3  # 3(xyz) + 3(RGB)

        # Encoder layers
        self.sa1 = MultiScaleSetAbstraction(1024, [0.1, 0.2], [16, 32], in_channel, [64, 64, 128])
        self.sa2 = MultiScaleSetAbstraction(512, [0.2, 0.4], [16, 32], 259, [128, 128, 256])
        self.sa3 = MultiScaleSetAbstraction(128, [0.4, 0.8], [16, 32], 515, [256, 256, 512])

        # Geometric feature extraction
        self.geometric1 = GeometricFeatureExtraction(128 * 2)
        self.geometric2 = GeometricFeatureExtraction(256 * 2)
        self.geometric3 = GeometricFeatureExtraction(512 * 2)

        # 预训练任务的头部
        # 1. 点云重建头
        self.reconstruction_head = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 3, 1)  # 输出xyz坐标
        )

        # 2. 旋转预测头
        self.rotation_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # 输出四元数
        )

    def encode(self, xyz, colors):
        """编码器前向传播"""
        # 特征处理
        pos_enc = self.bri_enc(xyz)
        colors = colors.transpose(1, 2)
        color_features = self.color_encoder(colors, xyz)
        fused_features = self.feature_fusion(pos_enc, color_features)

        # 编码器前向传播
        l1_xyz, l1_features = self.sa1(xyz, fused_features)
        l1_features = self.geometric1(l1_features, l1_xyz)

        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        l2_features = self.geometric2(l2_features, l2_xyz)

        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)
        l3_features = self.geometric3(l3_features, l3_xyz)

        return l3_xyz, l3_features

    def forward(self, xyz, colors, rotated_xyz=None):
        """
        xyz: [B, N, 3] 原始点云
        colors: [B, N, 3] RGB颜色
        rotated_xyz: [B, N, 3] 旋转后的点云(用于旋转预测任务)
        """
        # 对原始点云进行编码
        _, features = self.encode(xyz, colors) #B, D, N]

        # 重建任务
        reconstruction = self.reconstruction_head(features)

        # 如果提供了旋转点云，进行旋转预测
        rotation_pred = None
        if rotated_xyz is not None:
            _, rotated_features = self.encode(rotated_xyz, colors)
            # 使用全局最大池化得到全局特征
            global_features = torch.max(rotated_features, dim=2)[0]  # [B, 1024]
            rotation_pred = self.rotation_head(global_features)

        return reconstruction, rotation_pred


def pretrain(model, train_loader, epochs=10, device='cuda'):
    """
    预训练函数
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            # 获取批次数据
            xyz = batch['xyz'].to(device)  # [B, N, 3]
            features = batch['colors'].to(device)  # [B, N, 3]

            batch_size = xyz.shape[0]

            # 生成随机旋转矩阵
            rotation_matrix = random_rotation_matrix(batch_size).to(device)  # [B, 3, 3]

            # 需要调整xyz的形状以进行批量矩阵乘法
            # 将点云数据reshape为[B, N, 3]
            xyz_reshaped = xyz.view(batch_size, -1, 3)  # 确保形状是[B, N, 3]

            # 应用旋转
            rotated_xyz = torch.bmm(xyz_reshaped, rotation_matrix)  # [B, N, 3]

            # 获取四元数表示
            quaternion_gt = rotation_matrix_to_quaternion(rotation_matrix)  # [B, 4]

            # 前向传播
            optimizer.zero_grad()

            # 获取模型预测
            reconstructed_xyz, predicted_quaternion = model(rotated_xyz, features)
            # 调整重建输出的维度以匹配输入
            reconstructed_xyz = reconstructed_xyz.transpose(1, 2)  # 从 [B, 3, N] 变为 [B, N, 3]
            # 计算重建损失
            reconstruction_loss = F.mse_loss(reconstructed_xyz, xyz)

            # 计算旋转预测损失
            rotation_loss = F.mse_loss(predicted_quaternion, quaternion_gt)

            # 总损失
            loss = reconstruction_loss + rotation_loss

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 打印每个epoch的平均损失
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.4f}')

    return model


import torch
import math


def random_rotation_matrix(batch_size=1):
    """
    生成随机旋转矩阵
    返回: [batch_size, 3, 3] 的旋转矩阵
    """
    # 随机生成欧拉角
    theta = torch.rand(batch_size, 3) * 2 * math.pi  # 随机角度 [0, 2π]

    # 分别计算三个轴的旋转矩阵
    cos_x, sin_x = torch.cos(theta[:, 0]), torch.sin(theta[:, 0])
    cos_y, sin_y = torch.cos(theta[:, 1]), torch.sin(theta[:, 1])
    cos_z, sin_z = torch.cos(theta[:, 2]), torch.sin(theta[:, 2])

    # 创建绕x轴的旋转矩阵
    R_x = torch.zeros(batch_size, 3, 3)
    R_x[:, 0, 0] = 1
    R_x[:, 1, 1] = cos_x
    R_x[:, 1, 2] = -sin_x
    R_x[:, 2, 1] = sin_x
    R_x[:, 2, 2] = cos_x

    # 创建绕y轴的旋转矩阵
    R_y = torch.zeros(batch_size, 3, 3)
    R_y[:, 0, 0] = cos_y
    R_y[:, 0, 2] = sin_y
    R_y[:, 1, 1] = 1
    R_y[:, 2, 0] = -sin_y
    R_y[:, 2, 2] = cos_y

    # 创建绕z轴的旋转矩阵
    R_z = torch.zeros(batch_size, 3, 3)
    R_z[:, 0, 0] = cos_z
    R_z[:, 0, 1] = -sin_z
    R_z[:, 1, 0] = sin_z
    R_z[:, 1, 1] = cos_z
    R_z[:, 2, 2] = 1

    # 组合旋转矩阵 R = R_z @ R_y @ R_x
    R = torch.bmm(torch.bmm(R_z, R_y), R_x)

    return R


def rotation_matrix_to_quaternion(R):
    """
    将3x3旋转矩阵转换为四元数
    输入: R [batch_size, 3, 3] 旋转矩阵
    返回: q [batch_size, 4] 四元数 [w, x, y, z]
    """
    batch_size = R.shape[0]
    q = torch.zeros(batch_size, 4)

    # 计算四元数的w分量
    w = torch.sqrt(1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]) / 2
    w4 = 4 * w

    # 计算四元数的x, y, z分量
    x = (R[:, 2, 1] - R[:, 1, 2]) / w4
    y = (R[:, 0, 2] - R[:, 2, 0]) / w4
    z = (R[:, 1, 0] - R[:, 0, 1]) / w4

    # 组合四元数
    q[:, 0] = w  # w
    q[:, 1] = x  # x
    q[:, 2] = y  # y
    q[:, 3] = z  # z

    return q


def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵
    输入: q [batch_size, 4] 四元数 [w, x, y, z]
    返回: R [batch_size, 3, 3] 旋转矩阵
    """
    batch_size = q.shape[0]

    # 归一化四元数
    q = F.normalize(q, p=2, dim=1)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # 计算旋转矩阵的元素
    R = torch.zeros(batch_size, 3, 3)

    R[:, 0, 0] = 1 - 2 * y * y - 2 * z * z
    R[:, 0, 1] = 2 * x * y - 2 * w * z
    R[:, 0, 2] = 2 * x * z + 2 * w * y

    R[:, 1, 0] = 2 * x * y + 2 * w * z
    R[:, 1, 1] = 1 - 2 * x * x - 2 * z * z
    R[:, 1, 2] = 2 * y * z - 2 * w * x

    R[:, 2, 0] = 2 * x * z - 2 * w * y
    R[:, 2, 1] = 2 * y * z + 2 * w * x
    R[:, 2, 2] = 1 - 2 * x * x - 2 * y * y

    return R


# 辅助函数：Chamfer Distance 损失

class ChamferDistance(nn.Module):
    def forward(self, x, y):
        """
        x: [B, N, 3]
        y: [B, N, 3]
        """
        x = x.unsqueeze(2)  # [B, N, 1, 3]
        y = y.unsqueeze(1)  # [B, 1, N, 3]

        dist = torch.sum((x - y) ** 2, dim=-1)  # [B, N, N]

        min_dist_x = torch.min(dist, dim=2)[0]  # [B, N]
        min_dist_y = torch.min(dist, dim=1)[0]  # [B, N]

        chamfer_dist = torch.mean(min_dist_x) + torch.mean(min_dist_y)

        return chamfer_dist


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
