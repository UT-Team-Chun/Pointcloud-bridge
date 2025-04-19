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
from models.spg import SuperpointGraph
from models.RandLANet import RandLANet
from models.pointnet import PointNetSeg
from models.DGCNN import DGCNN
from models.PointTransformerV3 import PointTransformerV3
import pandas as pd
import matplotlib
import gc
import psutil
# 导入所需的工具和数据集类
from utils.BriPCDMulti_new import BriPCDMulti
from utils.logger_config import initialize_logger, get_logger
from train_MulSca_BriStruNet_CB import AverageMeter  # 导入 AverageMeter

# 设置更好的字体和风格，以适合SCI论文的可视化效果
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.minor.width'] = 0.6
plt.rcParams['ytick.minor.width'] = 0.6

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

def evaluate_model(model, model_name, dataloader, device, num_classes=5):
    """评估模型的各项性能指标"""
    model = model.to(device)
    model.eval()

    # 获取数据加载器中的点数信息 (假设所有样本点数相同)
    # 注意：BriPCDMulti 可能返回不同数量的点，这里取第一个样本的点数作为参考
    try:
        first_batch = next(iter(dataloader))
        # 确保 points 键存在且不为空
        if 'points' in first_batch and first_batch['points'].numel() > 0:
             # B, N, C -> N = points per sample
            num_points = first_batch['points'].shape[1]
            in_channels = first_batch['points'].shape[2] # 通常是 3 (xyz) 或者 6 (xyz + color/normal)
            # 检查是否有颜色信息
            if 'colors' in first_batch and first_batch['colors'] is not None and first_batch['colors'].numel() > 0:
                in_channels += first_batch['colors'].shape[2] # 通常是 3 (rgb)
            print(f"Detected num_points: {num_points}, in_channels: {in_channels}")
        else:
             # 如果没有 points 数据，或者 points 为空，则设置默认值或引发错误
             print("Warning: Could not determine num_points from the first batch. Using default 4096.")
             num_points = 4096 # 回退到默认值
             in_channels = 3 # 假设只有 xyz

    except StopIteration:
        print("Error: Dataloader is empty.")
        return None # 或者返回错误指示
    except Exception as e:
        print(f"Error getting data shape from dataloader: {e}. Using default 4096 points.")
        num_points = 4096 # 回退到默认值
        in_channels = 3

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
    num_batches_inference = 0 # 用于推理计时的批次数

    # 使用从数据加载器获取的真实数据批次进行测试
    # 这里我们只取一个批次进行测试，因为主要目的是测量单次前向传播时间
    test_batch = first_batch # 使用前面获取的第一个批次
    points = test_batch['points'].to(device) # [B, N, 3/6]
    colors = test_batch.get('colors', None) # [B, N, 3] or None
    if colors is not None:
        colors = colors.to(device)

    # 准备模型输入 (不同模型可能需要不同格式)
    # 假设模型需要 (xyz, features)
    xyz = points[:, :, :3] # 取前3维作为坐标 [B, N, 3]
    features = points[:, :, 3:] # 取后面的维度作为特征，如果点云包含颜色/法线的话
    if colors is not None:
         # 如果单独提供了颜色，且原始点云只有xyz，则使用颜色作为特征
         if features.shape[2] == 0:
             features = colors
         else: # 如果原始点云也有特征，则拼接颜色特征
             features = torch.cat((features, colors), dim=2) # [B, N, C_features + C_colors]

    # 确保特征维度不为0，如果为0，则可能只用xyz
    if features.shape[2] == 0:
        # 对于只需要坐标的模型
        model_input = (xyz,)
        print(f"{model_name}: Using only XYZ coordinates as input.")
    else:
        # 对于需要坐标和特征的模型
        model_input = (xyz, features)
        print(f"{model_name}: Using XYZ coordinates and features (dim={features.shape[2]}) as input.")


    batch_size = xyz.shape[0]

    # 预热GPU
    try:
        with torch.no_grad():
            for _ in range(10):
                _ = model(*model_input) # 使用解包传递参数
    except Exception as e:
        print(f"Error during GPU warmup for {model_name}: {e}")
        # 可以选择跳过此模型或返回错误
        return {**results, 'error': f"GPU warmup failed: {e}"} # 返回包含错误信息的结果

    # 测量推理时间
    try:
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            inference_runs = 10 # 多次运行以求平均
            for _ in range(inference_runs):
                _ = model(*model_input) # 使用解包传递参数
                num_batches_inference += 1
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

        total_time = end_time - start_time
        # 计算每次推理的平均时间 (ms)
        avg_time_ms = (total_time / inference_runs) * 1000
        results['inference_time_ms'] = avg_time_ms

        # 计算每秒处理的点数
        points_per_second = (batch_size * num_points) / (total_time / inference_runs)
        results['points_per_second'] = points_per_second

    except Exception as e:
        print(f"Error during inference measurement for {model_name}: {e}")
        return {**results, 'error': f"Inference failed: {e}"} # 返回包含错误信息的结果

    # 测量GPU内存使用
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
            _ = model(*model_input) # 使用解包传递参数
            results['gpu_memory_usage_mb'] = torch.cuda.max_memory_allocated() / 1024**2
        except Exception as e:
            print(f"Error measuring GPU memory for {model_name}: {e}")
            results['gpu_memory_usage_mb'] = -1 # 表示测量失败

    # 测量CPU内存使用
    process = psutil.Process(os.getpid())
    results['cpu_memory_usage_mb'] = process.memory_info().rss / 1024**2

    # 训练一个小epoch（或几个批次），测量训练时间
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 使用标准的 CrossEntropyLoss 进行时间测量
    criterion = nn.CrossEntropyLoss()

    # 只训练部分批次，以加快评估速度
    num_train_batches = min(10, len(dataloader))
    start_time = time.time()
    actual_train_batches = 0 # 记录实际完成的批次

    try:
        for i, batch in enumerate(dataloader):
            if i >= num_train_batches:
                break

            points = batch['points'].to(device) # [B, N, 3/6]
            colors = batch.get('colors', None) # [B, N, 3] or None
            labels = batch['labels'].to(device) # [B, N]

            if colors is not None:
                 colors = colors.to(device)

            # 准备模型输入 (同推理部分)
            xyz = points[:, :, :3]
            features = points[:, :, 3:]
            if colors is not None:
                 if features.shape[2] == 0:
                      features = colors
                 else:
                      features = torch.cat((features, colors), dim=2)

            if features.shape[2] == 0:
                 model_input = (xyz,)
            else:
                 model_input = (xyz, features)

            optimizer.zero_grad()
            outputs = model(*model_input) # 使用解包传递参数 # [B, N, C] 或 [B, C, N]

            # 确保输出和标签形状匹配 [B, N, C] vs [B, N]
            if outputs.dim() == 3 and outputs.shape[0] == labels.shape[0]:
                # Case 1: [B, N, C] - 标准语义分割输出
                if outputs.shape[1] == labels.shape[1]:
                    B, N, C = outputs.shape
                    outputs_for_loss = outputs.reshape(-1, C)
                    labels_for_loss = labels.reshape(-1)
                # Case 2: [B, C, N] - 有些模型会输出这种格式
                elif outputs.shape[2] == labels.shape[1]:
                    B, C, N = outputs.shape
                    outputs_for_loss = outputs.permute(0, 2, 1).reshape(-1, C) # 转换为 [B*N, C]
                    labels_for_loss = labels.reshape(-1)
                else:
                    print(f"Warning: Output shape {outputs.shape} mismatch with label shape {labels.shape} for {model_name}. Skipping loss calculation for this batch.")
                    continue # 跳过这个批次
            else:
                print(f"Warning: Unexpected output dimension ({outputs.dim()}) or batch size mismatch for {model_name}. Skipping loss calculation.")
                continue # 跳过这个批次

            loss = criterion(outputs_for_loss, labels_for_loss)
            loss.backward()
            optimizer.step()
            actual_train_batches += 1 # 成功完成一个批次

    except Exception as e:
        print(f"Error during training time measurement for {model_name}: {e}")
        results['training_time_per_epoch_s'] = -1 # 表示测量失败
    else:
        end_time = time.time()
        if actual_train_batches > 0:
            epoch_time = end_time - start_time
            # 估算完整epoch的训练时间
            full_epoch_time = epoch_time * (len(dataloader) / actual_train_batches) if actual_train_batches > 0 else 0
            results['training_time_per_epoch_s'] = full_epoch_time
        else:
            print(f"Warning: No batches were successfully trained for {model_name}.")
            results['training_time_per_epoch_s'] = 0 # 没有成功训练的批次

    # 清理
    del points, colors, labels, xyz, features, outputs, test_batch, first_batch, model_input
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
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.3)
    
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
    
    # 初始化日志记录器
    log_dir = Path('logs/eva_logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = initialize_logger(log_dir)
    logger.info("Starting model evaluation script.")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 数据集和模型参数配置 (使用真实数据)
    data_config = {
        'val_dir': 'data/all/val', # 使用验证集进行评估
        'num_points': 4096,       # 
        'block_size': 1.0,
        'sample_rate': 0.4
    }
    batch_size = 4  # 使用较小的批次大小进行评估
    num_workers = 0 # 在Windows上或调试时设为0，避免多进程问题
    num_classes = 5 # 根据你的任务设定

    logger.info(f"Data configuration: {data_config}")
    logger.info(f"Batch size: {batch_size}, Num workers: {num_workers}")


    # 创建真实数据加载器 (使用验证集)
    try:
        eval_dataset = BriPCDMulti(
            data_dir=data_config['val_dir'],
            num_points=data_config['num_points'],
            block_size=data_config['block_size'],
            sample_rate=data_config['sample_rate'],
            # use_normals=data_config['use_normals'], # 如果需要法线，取消注释
            logger=get_logger(),
            transform=False # 验证/评估时通常不进行数据增强
        )

        if len(eval_dataset) == 0:
            logger.error(f"No data found in {data_config['val_dir']}. Please check the path and data.")
            return # 没有数据无法继续

        dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False, # 评估时不需要打乱
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True # 确保所有批次大小一致，便于测量
        )
        logger.info(f"Successfully loaded evaluation dataset from {data_config['val_dir']}. Size: {len(eval_dataset)}")

    except FileNotFoundError:
        logger.error(f"Data directory not found: {data_config['val_dir']}")
        return
    except Exception as e:
        logger.error(f"Error creating dataset or dataloader: {e}")
        return

    # 获取输入通道数 (基于数据配置)
    # d_in = 3 # xyz
    # if data_config['use_color']:
    #     d_in += 3
    # if data_config['use_normals']:
    #     d_in += 3
    # 暂时从数据中动态获取，更加鲁棒
    first_batch_check = next(iter(dataloader))
    d_in = 3 # Start with xyz
    if 'colors' in first_batch_check and first_batch_check['colors'] is not None:
        d_in += first_batch_check['colors'].shape[-1]
    # if 'normals' in first_batch_check and first_batch_check['normals'] is not None: # 如果加载了法线
    #     d_in += first_batch_check['normals'].shape[-1]
    logger.info(f"Determined model input channels (d_in): {d_in}")


    # 修改模型评估列表，使用完整的模型进行评估
    models = []

    # 依次添加各个模型，使用try-except确保程序不会因某个模型初始化失败而中断
    try:
        # PointNetSeg 通常需要 (xyz, features) 或只有 (xyz)
        # 确认 PointNetSeg 实现是否接受 features
        models.append((PointNetSeg(num_classes), "PointNet")) # 假设它只用 xyz 或会自动处理
        logger.info("成功初始化 PointNet")
    except Exception as e:
        logger.error(f"初始化 PointNet 失败: {e}")

    try:
        # DGCNN 通常需要 (xyz, features) 或 (xyz)
        # 确认 DGCNN 实现
        models.append((DGCNN(num_classes, k=32), "DGCNN")) # 假设它只用 xyz 或会自动处理, k是超参数
        logger.info("成功初始化 DGCNN")
    except Exception as e:
        logger.error(f"初始化 DGCNN 失败: {e}")

    try:
        # PointNet2 通常需要 (xyz, features)，features 可以是颜色/法线
        # 确认 PointNet2 实现
        models.append((PointNet2(num_classes), "PointNet2")) # 假设它需要 (xyz, features)
        logger.info("成功初始化 PointNet2")
    except Exception as e:
        logger.error(f"初始化 PointNet2 失败: {e}")

    try:
        # SPG 输入较特殊，通常需要超点图构建过程，直接评估可能不适用或需要适配
        # 这里的 SuperpointGraph 可能是一个简化版本或需要特定输入
        models.append((SuperpointGraph(num_classes, input_channels=d_in,
                                       superpoint_size=50, # 超点大小
                                       emb_dims=1024           # 全局特征维度
                                       ), "SPG"))
        #logger.warning("SPG 模型评估已暂时禁用，因其输入结构特殊。")
        logger.info("成功初始化 SPG")
    except Exception as e:
        logger.error(f"初始化 SPG 失败: {str(e)}")

    try:
        # RandLA-Net 需要指定 d_in
        models.append((RandLANet(num_classes, d_in=d_in), "RandLA-Net"))
        logger.info("成功初始化 RandLA-Net")
    except Exception as e:
        logger.error(f"初始化 RandLA-Net 失败: {str(e)}")
    
    try:
        # PointTransformerV3 通常需要 (xyz, features)
        models.append((PointTransformerV3(num_classes, d_in=d_in,embed_dim=384, 
                                          depth=12,  # 深度
                                          num_heads=6, # 多头注意力机制
                                          use_flash=False # 启用Flash Attention加速
                                          ), "PointTransformerV3"))
        logger.info("成功初始化 PointTransformerV3")
    except Exception as e:
        logger.error(f"初始化 PointTransformerV3 失败: {str(e)}")

    try:
        # EnhancedPointNet2 (BridgeSeg) 通常需要 (xyz, features)
        models.append((EnhancedPointNet2(num_classes=num_classes), "BridgeSeg"))
        logger.info("成功初始化 BridgeSeg")
    except Exception as e:
        logger.error(f"初始化 BridgeSeg 失败: {e}")


    # 评估所有模型
    results = []

    logger.info("=" * 60)
    logger.info("Starting comprehensive model evaluation")
    logger.info("=" * 60)

    for model, model_name in tqdm(models, desc="Evaluating models"):
        logger.info(f"Evaluating {model_name}...")
        try:
            # 清理内存，为下一个模型评估做准备
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model_results = evaluate_model(model, model_name, dataloader, device,
                                          num_classes=num_classes)

            if model_results: # 检查 evaluate_model 是否成功返回结果
                 if 'error' in model_results:
                     logger.error(f"Evaluation failed for {model_name}: {model_results['error']}")
                 else:
                     results.append(model_results)
                     logger.info(f"Completed evaluation of {model_name}")
                     logger.info(f"Parameters: {model_results['parameters']:,}")
                     logger.info(f"Inference time: {model_results['inference_time_ms']:.2f} ms")
                     logger.info(f"GPU memory usage: {model_results['gpu_memory_usage_mb']:.2f} MB")
                     logger.info(f"Training time per epoch: {model_results['training_time_per_epoch_s']:.2f} s")
                     logger.info(f"Points per second: {model_results.get('points_per_second', 'N/A'):,.0f}")
            else:
                 logger.warning(f"Evaluation skipped or failed for {model_name}, no results returned.")

        except Exception as e:
            logger.error(f"Critical error evaluating {model_name}: {str(e)}", exc_info=True) # 记录详细的回溯信息

    # 保存结果到CSV
    if results:
        # 在保存前过滤掉包含错误的条目
        valid_results = [r for r in results if 'error' not in r]
        if not valid_results:
            logger.warning("No valid results were generated after filtering errors.")
        else:
            df = save_results_to_csv(valid_results) # 使用过滤后的结果

            # 创建可视化
            create_visualizations(df)

            # 打印结果表格
            logger.info("=" * 60)
            logger.info("Model Performance Comparison Summary:")
            logger.info("=" * 60)
            logger.info(df.to_string(index=False))

            logger.info("Evaluation complete! Results saved to CSV and visualizations generated.")
    else:
        logger.warning("No results were generated. Please check the logs for errors.")

if __name__ == "__main__":
    main()