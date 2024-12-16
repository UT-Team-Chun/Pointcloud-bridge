# models/enhanced_pointnet2.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_modules import BoundaryAwareModule, EnhancedAttentionModule, \
    GeometricFeatureExtraction, EnhancedPositionalEncoding
from .pointnet2_utils import FeaturePropagation, SetAbstraction, MultiScaleSetAbstraction


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
        input_ch=6
        self.pos_encoding = EnhancedPositionalEncoding(input_ch,4,64,)

        in_chanel = input_ch + 6 # 3(xyz) + 3(RGB)
        # Encoder
        #self.sa1 = SetAbstraction(1024, 0.1, 32, in_chanel, [64, 64, 128]) #6+64
        #self.sa2 = SetAbstraction(256, 0.2, 32, 131, [128, 128, 256]) #128*2
        #self.sa3 = SetAbstraction(64, 0.4, 32, 259, [256, 256, 512]) #256*2

        # 1st layer: input = 3(xyz) + 3(RGB) + 64(pos_encoding) = 70
        self.sa1 = MultiScaleSetAbstraction(1024, [0.1, 0.2],[16, 32], 6, [64, 64, 128])
        # 2nd layer: input = 128*2 (Multi-scale connection)
        self.sa2 = MultiScaleSetAbstraction(256,[0.2, 0.4],[16, 32], 259,[128, 128, 256])
        self.sa3 = MultiScaleSetAbstraction(64,[0.4, 0.8],[16, 32], 515,[256, 256, 512])
        #  npoint=1024,radius_list=[0.1, 0.2], nsample_list=[16, 32], in_channel=6+64, mlp=[64, 64, 128]

        # attention module
        self.attention1 = EnhancedAttentionModule(128*2) #128 * 2
        self.attention2 = EnhancedAttentionModule(256*2)
        self.attention3 = EnhancedAttentionModule(512*2)

        # Geometric feature extraction
        self.geometric1 = GeometricFeatureExtraction(128 * 2)
        self.geometric2 = GeometricFeatureExtraction(256 * 2)
        self.geometric3 = GeometricFeatureExtraction(512 * 2)

        # Boundary aware modules
        self.boundary1 = BoundaryAwareModule(128*2)  # 256 channels * 2
        self.boundary2 = BoundaryAwareModule(256*2)  # 512 channels * 2
        self.boundary3 = BoundaryAwareModule(512*2)  # 1024 channels * 2

        # Decoder
        self.fp3 = FeaturePropagation(1536, [512, 256])  # multi: 512*2 + 256*2 ,1536 ; 256 + 256*2 ,768
        self.fp2 = FeaturePropagation(512, [256, 128])  # multi:  256 + 128*2 ,768; 128 + 128*2 ,384
        self.fp1 = FeaturePropagation(128, [128, 128, 128]) # multi:  128 + 64 128, 384; 128 + 64 128

        # Final layers
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, colors):

        """
        xyz: [B, N, 3]
        points: [B, N, 32] (RGB)
        """

        # Add positional encoding
        pos_enc = self.pos_encoding(xyz) # [B, 64, N]
        # Change the order of dimensions
        colors = colors.transpose(1, 2)  # [B, 3, N]
        features = torch.cat([pos_enc, colors], dim=1)  # Merge Features: [B, 67, N]

        # Encoder with multi-scale feature extraction
        l1_xyz, l1_points = self.sa1(xyz, colors)   #[B,70,N] -> [B, 128, N]
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


if __name__ == "__main__":

    # 设置随机种子保证可复现性
    torch.manual_seed(42)

    # 创建测试数据
    batch_size = 2
    num_points = 1024

    # 创建随机输入数据
    xyz = torch.randn(batch_size, num_points, 3)
    features = torch.randn(batch_size, num_points, 3)

    # 实例化模型
    model = EnhancedPointNet2()  # 替换为你的实际模型名
    model.eval()

    # 1. 基础功能测试
    print("=" * 50)
    print("基础功能测试")
    print(f"输入 xyz shape: {xyz.shape}")
    print(f"输入 features shape: {features.shape}")

    try:
        output = model(xyz, features)
        print(f"输出 shape: {output.shape}")
        print("模型前向传播测试通过!")
    except Exception as e:
        print(f"模型运行出错: {str(e)}")

    # 2. 模型信息统计
    print("\n" + "=" * 50)
    print("模型信息统计")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    #print(f"模型结构:\n{model}")

    # # 3. Tensorboard可视化
    # print("\n" + "=" * 50)
    # print("生成Tensorboard可视化文件")
    #
    # writer = SummaryWriter('runs/model_visualization')
    # writer.add_graph(model, (xyz, features))
    # writer.close()
    # print("Tensorboard文件已生成,使用 'tensorboard --logdir=runs' 查看")
    #
    # # 4. ONNX导出与可视化
    # print("\n" + "=" * 50)
    # print("ONNX模型导出")
    #
    # try:
    #     torch.onnx.export(model,  # 模型
    #                       (xyz, features),  # 模型输入
    #                       "model.onnx",  # 保存路径
    #                       input_names=['xyz', 'features'],  # 输入名
    #                       output_names=['output'],  # 输出名
    #                       dynamic_axes={'xyz': {0: 'batch_size'},  # 动态轴
    #                                     'features': {0: 'batch_size'},
    #                                     'output': {0: 'batch_size'}})
    #
    #     # 验证ONNX模型
    #     onnx_model = onnx.load("model.onnx")
    #     onnx.checker.check_model(onnx_model)
    #     print("ONNX模型导出成功且验证通过!")
    #     print("可使用Netron(https://netron.app)查看模型结构")
    # except Exception as e:
    #     print(f"ONNX导出失败: {str(e)}")

    # 7. 测试不同batch size的内存占用
    print("\n" + "=" * 50)
    print("内存占用测试")

    batch_sizes = [4, 16, 32, 64]
    for bs in batch_sizes:
        torch.cuda.empty_cache()  # 清空GPU缓存
        xyz_test = torch.randn(bs, num_points, 3)
        features_test = torch.randn(bs, num_points, 3)

        if torch.cuda.is_available():
            model = model.cuda()
            xyz_test = xyz_test.cuda()
            features_test = features_test.cuda()

            torch.cuda.reset_peak_memory_stats()
            output = model(xyz_test, features_test)
            memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # 转换为MB
            print(f"Batch size {bs}: 峰值显存占用 {memory:.2f} MB")

    print("\n测试完成!")
