import torch
import torch.nn as nn
from models.enhanced_pointnet2 import EnhancedPointNet2
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#tensorboard --logdir ./logs
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import datetime
import os
import torch.nn.functional as F
from models.enhanced_pointnet2 import EnhancedPointNet2
from utils.data_utils import BridgePointCloudDataset

def export_to_onnx():
    # 创建模型
    num_classes = 5
    model = EnhancedPointNet2(num_classes)
    model.eval()

    # 创建示例输入
    dummy_points = torch.randn(16, 4096, 3, dtype=torch.float32)
    dummy_colors = torch.randn(16, 4096, 3, dtype=torch.float32)
    writer = SummaryWriter('experiments/tensorboard')
    writer.add_graph(model, [dummy_points, dummy_colors])
    writer.close()

    # try:
    #     # 使用动态轴和简化的导出设置
    #     torch.onnx.export(
    #         model,
    #         (dummy_points, dummy_colors),
    #         "models/model.onnx",
    #         export_params=True,
    #         opset_version=14,  # 降低 opset 版本
    #         do_constant_folding=True,
    #         input_names=['points', 'colors'],
    #         output_names=['output'],
    #         dynamic_axes={
    #             'points': {0: 'batch_size'},
    #             'colors': {0: 'batch_size'},
    #             'output': {0: 'batch_size'}
    #         }
    #     )
    #     print("Successfully exported to ONNX")
    #
    #     # 验证导出的模型
    #     #import onnx
    #     ##onnx_model = onnx.load("models/model.onnx")
    #     #onnx.checker.check_model(onnx_model)
    #     #print("Successfully validated ONNX model")

    # except Exception as e:
    #     print(f"Error during export: {str(e)}")
    #     import traceback
    #     traceback.print_exc()


# 配置参数
config = {
    'num_points': 4096,
    'batch_size': 32,
    'num_workers': 4,
    'learning_rate': 0.001,
    'num_epochs': 1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def train():
    # 创建实验目录
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(f'experiments/exp_{timestamp}')
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 设置tensorboard和日志
    writer = SummaryWriter('experiments/tensorboard')

    # 设置设备
    device = torch.device(config['device'])
    # 创建数据加载器
    train_dataset = BridgePointCloudDataset(
        data_dir='data/train',
        num_points=config['num_points'],
        transform=True
    )
    val_dataset = BridgePointCloudDataset(
        data_dir='data/test',
        num_points=config['num_points'],
        transform=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # 创建模型
    num_classes = 5
    model = EnhancedPointNet2(num_classes).to(device)

    # 损失函数和优化器
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

    # 添加学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,
        T_mult=2,
        eta_min=1e-6
    )

    for epoch in range(config['num_epochs']):
        # 训练阶段
        model.eval()

        # 创建进度条
        total_samples = len(train_dataset)
        pbar = tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{config["num_epochs"]}] Training',
                    total=total_samples // config['batch_size'])

        for batch in pbar:
            points = batch['points'].to(device)
            colors = batch['colors'].to(device)
            labels = batch['labels'].to(device)

            #optimizer.zero_grad()
            #outputs = model(points, colors)
            #loss = criterion(outputs, labels)

            writer.add_graph(model, [points, colors])
            writer.close()
            print('saved the model')

            torch.onnx.export(
                model,
                (points, colors),
                "models/model.onnx",
                export_params=True,
                opset_version=14,  # 降低 opset 版本
                do_constant_folding=True,
                input_names=['points', 'colors'],
                output_names=['output'],
                dynamic_axes={
                    'points': {0: 'batch_size'},
                    'colors': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            print("Successfully exported to ONNX")







if __name__ == "__main__":
    train()
