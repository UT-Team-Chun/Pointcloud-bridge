import torch

from models.model import EnhancedPointNet2

config = {
    'transformer_config': {
        'trans_dim': 384,
        'depth': 12,
        'drop_path_rate': 0.1,
        'num_heads': 6,
        'encoder_dims': 384,
    },
    'num_group': 64,  # 移到外层
    'group_size': 32,  # 移到外层
    'num_points': 4096,
    'chunk_size': 4096,
    'overlap': 1024,
    'batch_size': 16,
    'num_workers': 0,
    'learning_rate': 0.001,
    'num_classes': 5,
    'num_epochs': 500,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def debug_model():
    # 设置随机种子保证可复现性
    torch.manual_seed(42)

    # 创建测试数据
    batch_size = 2
    num_points = 1024

    # 创建随机输入数据
    xyz = torch.randn(batch_size, num_points, 3)
    features = torch.randn(batch_size, num_points, 3)

    # 实例化模型
    model = EnhancedPointNet2()# 替换为你的实际模型名
    model.eval()

    # 1. 基础功能测试
    print("=" * 50)
    print("基础功能测试")
    print(f"输入 xyz shape: {xyz.shape}")
    print(f"输入 features shape: {features.shape}")

    try:
        output = model(xyz, features)
        print(f'output is {output}')
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
    # print(f"模型结构:\n{model}")

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

    batch_sizes = [4, 16]
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

if __name__ == "__main__":
    debug_model()