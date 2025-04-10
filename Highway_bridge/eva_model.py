import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import laspy
import time
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
from models.model import EnhancedPointNet2, PointNet2
from models.model import PointNetSeg, DGCNN
from models.spg import SuperpointGraph
from models.RandLANet import RandLANet
import pandas as pd
import matplotlib
import gc
import psutil

# 设置更好的字体和风格，以适合SCI论文的可视化效果
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.minor.width'] = 0.6
plt.rcParams['ytick.minor.width'] = 0.6

# 创建更大的点云数据集
class RandomPointCloudDataset(Dataset):
    def __init__(self, num_samples=1000, num_points=8192, num_classes=5):
        self.num_samples = num_samples
        self.num_points = num_points
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成随机点云数据
        xyz = torch.randn(self.num_points, 3)  # [N, 3]
        colors = torch.rand(self.num_points, 3)  # [N, 3]
        
        # 生成随机语义分割标签
        labels = torch.randint(0, self.num_classes, (self.num_points,))

        # 确保数据类型和形状正确
        xyz = xyz.float()
        colors = colors.float()
        labels = labels.long()

        return {
            'xyz': xyz,  # [N, 3]
            'colors': colors,  # [N, 3]
            'labels': labels  # [N]
        }

def get_model_size(model):
    """计算模型大小（MB）"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model(model, model_name, dataloader, device, num_classes=5, num_points=8192):
    """评估模型的各项性能指标"""
    model = model.to(device)
    model.eval()
    
    results = {
        'model_name': model_name,
        'parameters': count_parameters(model),
        'model_size_mb': get_model_size(model),
        'gpu_memory_usage_mb': 0,
        'cpu_memory_usage_mb': 0,
        'inference_time_ms': 0,
        'flops_g': 0,
        'training_time_per_epoch_s': 0
    }
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # 测量推理时间
    total_time = 0
    num_batches = 0
    
    # 创建一个用于推理测试的批次
    test_batch = next(iter(dataloader))
    xyz = test_batch['xyz'].to(device)
    features = test_batch['colors'].to(device)
    batch_size = xyz.shape[0]
    
    # 预热GPU
    with torch.no_grad():
        for _ in range(10):
            _ = model(xyz, features)
    
    # 测量推理时间
    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100):  # 多次测量以获得更准确的结果
            _ = model(xyz, features)
            num_batches += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_ms = (total_time / num_batches) * 1000
    results['inference_time_ms'] = avg_time_ms
    
    # 计算每秒处理的点数
    points_per_second = (batch_size * num_points * num_batches) / total_time
    results['points_per_second'] = points_per_second
    
    # 测量GPU内存使用
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _ = model(xyz, features)
        results['gpu_memory_usage_mb'] = torch.cuda.max_memory_allocated() / 1024**2
    
    # 测量CPU内存使用
    process = psutil.Process(os.getpid())
    results['cpu_memory_usage_mb'] = process.memory_info().rss / 1024**2
    
    # 训练一个小epoch，测量训练时间
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 只训练部分批次，以加快评估速度
    num_train_batches = min(10, len(dataloader))
    start_time = time.time()
    
    for i, batch in enumerate(dataloader):
        if i >= num_train_batches:
            break
            
        xyz = batch['xyz'].to(device)
        features = batch['colors'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(xyz, features)
        
        # 根据模型输出格式调整损失计算
        if outputs.dim() == 2:  # [B, C] 全局分类
            loss = criterion(outputs, labels[:, 0])
        else:  # [B, N, C] 点级分割
            B, N, C = outputs.shape
            
            # 修复批次大小不匹配问题 - 确保输出和标签形状一致
            if outputs.shape[0] != labels.shape[0] or outputs.shape[1] != labels.shape[1]:
                # 如果是转置的问题 (比如 [B, C, N] 而不是 [B, N, C])
                if outputs.shape[0] == labels.shape[0] and outputs.shape[2] == labels.shape[1]:
                    outputs = outputs.transpose(1, 2)
                    B, N, C = outputs.shape
                # 如果是维度不一致
                elif outputs.shape[1] == 1:
                    # 某些模型可能只返回全局特征，需要广播到所有点
                    outputs = outputs.expand(-1, labels.shape[1], -1)
                    B, N, C = outputs.shape
                
            outputs = outputs.reshape(-1, C)
            labels = labels.reshape(-1)
            loss = criterion(outputs, labels)
            
        loss.backward()
        optimizer.step()
    
    end_time = time.time()
    epoch_time = end_time - start_time
    
    # 估算完整epoch的训练时间
    full_epoch_time = epoch_time * (len(dataloader) / num_train_batches)
    results['training_time_per_epoch_s'] = full_epoch_time
    
    # 清理
    del xyz, features, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return results

def save_results_to_csv(results, output_path='model_performance_comparison.csv'):
    """将结果保存为CSV文件"""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    return df

def create_visualizations(df, output_dir='visualization_results'):
    """创建可视化图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置漂亮的可视化样式
    sns.set_style("whitegrid")
    matplotlib.rcParams['axes.edgecolor'] = '#333333'
    matplotlib.rcParams['axes.linewidth'] = 0.8
    matplotlib.rcParams['xtick.color'] = '#333333'
    matplotlib.rcParams['ytick.color'] = '#333333'
    
    # 颜色方案 - 使用低对比度颜色
    n_models = len(df)
    colors = sns.color_palette("pastel", n_colors=n_models)
    
    # 修复seaborn警告 - 1. 参数量比较图
    plt.figure(figsize=(5, 4), dpi=300)
    ax = sns.barplot(x='model_name', y='parameters', hue='model_name', data=df, palette=colors, legend=False)
    ax.set_ylabel('Number of Parameters')
    ax.set_xlabel('')
    plt.title('Model Size Comparison (Parameters)', fontweight='normal')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(pad=1.1)
    plt.savefig(os.path.join(output_dir, 'parameters_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. 推理时间比较图
    plt.figure(figsize=(5, 4), dpi=300)
    ax = sns.barplot(x='model_name', y='inference_time_ms', hue='model_name', data=df, palette=colors, legend=False)
    ax.set_ylabel('Inference Time (ms)')
    ax.set_xlabel('')
    plt.title('Inference Speed Comparison', fontweight='normal')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(pad=1.1)
    plt.savefig(os.path.join(output_dir, 'inference_time_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3. 训练时间比较图
    plt.figure(figsize=(5, 4), dpi=300)
    ax = sns.barplot(x='model_name', y='training_time_per_epoch_s', hue='model_name', data=df, palette=colors, legend=False)
    ax.set_ylabel('Training Time per Epoch (s)')
    ax.set_xlabel('')
    plt.title('Training Time Comparison', fontweight='normal')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(pad=1.1)
    plt.savefig(os.path.join(output_dir, 'training_time_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 4. 内存使用比较图
    plt.figure(figsize=(5, 4), dpi=300)
    ax = sns.barplot(x='model_name', y='gpu_memory_usage_mb', hue='model_name', data=df, palette=colors, legend=False)
    ax.set_ylabel('GPU Memory Usage (MB)')
    ax.set_xlabel('')
    plt.title('GPU Memory Consumption Comparison', fontweight='normal')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(pad=1.1)
    plt.savefig(os.path.join(output_dir, 'memory_usage_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 5. 雷达图比较所有指标
    # 标准化数据以便在雷达图中比较
    metrics = ['parameters', 'inference_time_ms', 'gpu_memory_usage_mb', 'training_time_per_epoch_s']
    df_radar = df.copy()
    
    # 对于参数，小值更好，所以我们反转标准化
    df_radar['parameters_norm'] = 1 - (df_radar['parameters'] / df_radar['parameters'].max())
    # 对于其他指标，小值也更好，所以也反转标准化
    df_radar['inference_norm'] = 1 - (df_radar['inference_time_ms'] / df_radar['inference_time_ms'].max())
    df_radar['memory_norm'] = 1 - (df_radar['gpu_memory_usage_mb'] / df_radar['gpu_memory_usage_mb'].max())
    df_radar['training_norm'] = 1 - (df_radar['training_time_per_epoch_s'] / df_radar['training_time_per_epoch_s'].max())
    
    # 绘制雷达图
    categories = ['Parameter\nEfficiency', 'Inference\nSpeed', 'Memory\nEfficiency', 'Training\nSpeed']
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(111, polar=True)
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    
    for i, model in enumerate(df_radar['model_name']):
        values = [df_radar.iloc[i]['parameters_norm'], 
                 df_radar.iloc[i]['inference_norm'],
                 df_radar.iloc[i]['memory_norm'],
                 df_radar.iloc[i]['training_norm']]
        values += values[:1]  # 闭合雷达图
        ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=model, color=colors[i % len(colors)])
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.25)
    
    ax.set_rlabel_position(0)
    ax.set_rticks([0.25, 0.5, 0.75])
    ax.set_rmax(1.0)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Model Performance Comparison', fontweight='normal', y=1.08)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def main():
    # 设置随机种子保证可复现性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 调整数据集和模型参数，避免内存和批次大小错误问题
    batch_size = 4  # 保持较小的批次大小
    num_points = 1024  # 减少点数量，避免内存问题
    num_classes = 5
    
    dataset = RandomPointCloudDataset(num_samples=50, num_points=num_points, num_classes=num_classes)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 不使用多进程，避免可能的问题
        drop_last=True
    )
    
    # 修改模型评估列表，使用完整的模型进行评估
    models = []
    
    # 依次添加各个模型，使用try-except确保程序不会因某个模型初始化失败而中断
    try:
        models.append((PointNetSeg(num_classes), "PointNet"))
        print("成功初始化 PointNet")
    except Exception as e:
        print(f"初始化 PointNet 失败: {e}")
    
    try:
        models.append((DGCNN(num_classes), "DGCNN"))
        print("成功初始化 DGCNN")
    except Exception as e:
        print(f"初始化 DGCNN 失败: {e}")
    
    try:
        models.append((PointNet2(num_classes), "PointNet2"))
        print("成功初始化 PointNet2")
    except Exception as e:
        print(f"初始化 PointNet2 失败: {e}")
        
    try:
        models.append((EnhancedPointNet2(num_classes=num_classes), "BridgeSeg"))
        print("成功初始化 BridgeSeg")
    except Exception as e:
        print(f"初始化 BridgeSeg 失败: {e}")
    
    try:
        # 使用正确实现的RandLANet
        models.append((RandLANet(num_classes, d_in=3), "RandLA-Net"))
        print("成功初始化 RandLA-Net")
    except Exception as e:
        print(f"初始化 RandLA-Net 失败: {str(e)}")
        
    try:
        # 添加新的SPG模型
        models.append((SuperpointGraph(num_classes, input_channels=3), "SPG"))
        print("成功初始化 SPG")
    except Exception as e:
        print(f"初始化 SPG 失败: {str(e)}")
    
    # 评估所有模型
    results = []
    
    print("\n" + "=" * 60)
    print("Starting comprehensive model evaluation")
    print("=" * 60)
    
    for model, model_name in tqdm(models, desc="Evaluating models"):
        print(f"\nEvaluating {model_name}...")
        try:
            model_results = evaluate_model(model, model_name, dataloader, device, 
                                          num_classes=num_classes, num_points=num_points)
            results.append(model_results)
            print(f"Completed evaluation of {model_name}")
            print(f"Parameters: {model_results['parameters']:,}")
            print(f"Inference time: {model_results['inference_time_ms']:.2f} ms")
            print(f"GPU memory usage: {model_results['gpu_memory_usage_mb']:.2f} MB")
            print(f"Training time per epoch: {model_results['training_time_per_epoch_s']:.2f} s")
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
    
    # 保存结果到CSV
    if results:
        df = save_results_to_csv(results)
        
        # 创建可视化
        create_visualizations(df)
        
        # 打印结果表格
        print("\n" + "=" * 60)
        print("Model Performance Comparison Summary:")
        print("=" * 60)
        print(df.to_string(index=False))
        
        print("\nEvaluation complete! Results saved to CSV and visualizations generated.")
    else:
        print("No results were generated. Please check for errors.")

if __name__ == "__main__":
    main()