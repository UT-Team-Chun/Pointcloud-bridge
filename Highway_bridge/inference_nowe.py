import logging
import os
from pathlib import Path

import laspy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.font_manager as fm
from matplotlib import rcParams

# 设置更好的绘图参数
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# 创建科学风格的颜色映射
sci_colors = ['#4878d0', '#ee854a', '#6acc64', '#d65f5f', '#956cb4', 
              '#8c613c', '#dc7ec0', '#797979', '#d5bb67', '#82c6e2']


def setup_logging(log_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir +'/testing.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main():
    # 配置参数
    config = {
        'num_points': 4096,
        'chunk_size': 4096,
        'overlap': 1024,
        'batch_size': 16,
        'num_workers': 6,
        'learning_rate': 0.001,
        'num_classes': 5,
        'num_epochs': 500,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'exp_dir' : 'experiments/exp_041922_DGCNN_DS1_ALL_0',
        'val_dir': 'data/CB/section/val',
    }
    
    exp_dir= config['exp_dir']
    checkpoint_path = os.path.join(exp_dir, 'best_model.pth')
    logger = setup_logging(exp_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    model_name = 'DGCNN'

    # 定义类名称
    class_names = ['Background', 'Pier', 'Girder', 'Deck', 'Parapet']

    from experiments.exp_041922_DGCNN_DS1_ALL_0.utils.BriPCDMulti_new import BriPCDMulti
    from experiments.exp_041922_DGCNN_DS1_ALL_0.models.model import EnhancedPointNet2
    from experiments.exp_041922_DGCNN_DS1_ALL_0.models.pointnet2 import PointNet2
    from models.DGCNN import DGCNN
    from torch.utils.data import DataLoader
    from utils.simpdataset import SimplePointCloudDataset

    test_dataset = SimplePointCloudDataset(
        data_dir=config['val_dir'],
        num_points=config['num_points'],
        steps_per_file=20, # 每个文件每个epoch采样50次
        #block_size=1.0,
        #sample_rate=0.4,
        logger=logger
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # 加载模型
    #model = EnhancedPointNet2(config['num_classes']).to(device)
    model=DGCNN(config['num_classes'], k=8).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'Loaded model from {checkpoint_path}')
    else:
        logger.error(f'No checkpoint found at {checkpoint_path}')
        return
    
    model.eval()
    
    # 创建输出目录
    output_dir = Path(exp_dir) / 'predicted_las'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建可视化输出目录
    viz_dir = Path(exp_dir) / 'visualization'
    viz_dir.mkdir(parents=True, exist_ok=True)

    val_acc = AverageMeter()
    confusion_matrix = torch.zeros(config['num_classes'], config['num_classes']).to(device)
    class_correct = torch.zeros(config['num_classes']).to(device)
    class_total = torch.zeros(config['num_classes']).to(device)

    # 创建一个字典来存储每个文件的预测结果
    file_predictions = {}
    # 创建一个字典来存储每个文件的评估指标
    file_metrics = {}

    with torch.no_grad():
        all_preds = []
        all_labels = []

        pbar = tqdm(test_loader, total=len(test_loader), desc='Processing files', position=0, leave=True)

        for batch in pbar:
            points = batch['points'].to(device, non_blocking=True)
            colors = batch['colors'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            file_names = batch['file_name']  # 获取文件名
            original_points = batch['original_points']
            original_colors = batch['original_colors']
            indices = batch['indices']

            outputs = model(points, colors)

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
            
            outputs = outputs_for_loss
            labels = labels_for_loss

            # 计算准确率
            pred = outputs.max(1)[1]
            acc_1 = pred.eq(labels).float().mean()


            # 更新统计
            val_acc.update(acc_1.item())

            all_preds.append(pred)
            all_labels.append(labels)

            # 将预测结果存储到对应文件，并按文件更新混淆矩阵
            for i in range(len(file_names)):
                fname = file_names[i]
                
                # 从缓存文件名中提取原始H5文件名
                # 格式为: filename_cache_id_block_i.npz
                parts = fname.split('_')
                if len(parts) >= 3 and parts[-2] == 'block':
                    # 提取原始文件名（去掉缓存ID和块信息）
                    cache_id_index = -3  # 缓存ID位置
                    original_filename = '_'.join(parts[:cache_id_index])
                    
                    # 移除.npz后缀（如果有）
                    if original_filename.endswith('.npz'):
                        original_filename = original_filename[:-4]
                else:
                    # 如果不符合预期格式，使用整个文件名
                    original_filename = fname
                    if original_filename.endswith('.npz'):
                        original_filename = original_filename[:-4]
                
                if original_filename not in file_predictions:
                    file_predictions[original_filename] = {
                        'points': [],
                        'colors': [],
                        'predictions': [],
                        'indices': [],
                        'true_labels': []
                    }
                    # 为每个文件创建混淆矩阵
                    file_metrics[original_filename] = {
                        'confusion_matrix': torch.zeros(config['num_classes'], config['num_classes']).to(device)
                    }

                file_predictions[original_filename]['points'].append(original_points[i])
                file_predictions[original_filename]['colors'].append(original_colors[i])
                file_predictions[original_filename]['predictions'].append(pred[i].cpu())
                file_predictions[original_filename]['indices'].append(indices[i])
                file_predictions[original_filename]['true_labels'].append(labels[i].cpu())

                # 更新该文件的混淆矩阵
                for t, p in zip(labels[i].view(-1), pred[i].view(-1)):
                    file_metrics[original_filename]['confusion_matrix'][t.long(), p.long()] += 1

            # 更新全局混淆矩阵
            for t, p in zip(labels.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            # 更新进度条
            pbar.set_postfix({'Val_Acc': f'{val_acc.avg * 100:.2f}%'})

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        total_acc = all_preds.eq(all_labels).float().mean()

        # 计算每个类别的准确率
        for i in range(config['num_classes']):
            mask = all_labels == i
            class_correct[i] = all_preds[mask].eq(all_labels[mask]).sum()
            class_total[i] = mask.sum()

        # 计算所有指标
        global_metrics = calculate_metrics(confusion_matrix)

        # 计算每个文件的指标
        for fname in file_metrics:
            file_metrics[fname].update(calculate_metrics(file_metrics[fname]['confusion_matrix']))

        # 创建保存指标数据的目录
        metrics_dir = Path(exp_dir) / 'metrics'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存指标数据为CSV
        save_metrics_to_csv(global_metrics, metrics_dir, 'global_metrics', class_names)
        
        # 保存每个文件的指标
        for fname, metrics in file_metrics.items():
            save_metrics_to_csv(metrics, metrics_dir, f'file_metrics_{fname}', class_names)
        
        # 记录全局结果
        logger.info("\n===== 全局评估结果 =====")
        logger.info(f"Mean IoU: {global_metrics['mIoU'] * 100:.2f}%")
        logger.info(f"Overall Accuracy: {global_metrics['OA'] * 100:.2f}%")
        logger.info(f"Mean Accuracy: {global_metrics['mAcc'] * 100:.2f}%")
        logger.info(f"Precision: {global_metrics['Precision'] * 100:.2f}%")
        logger.info(f"Recall: {global_metrics['Recall'] * 100:.2f}%")
        logger.info(f"F1 Score: {global_metrics['F1_score'] * 100:.2f}%")

        # 记录每个类别的IoU
        for i, class_iou in enumerate(global_metrics['IoU_per_class']):
            logger.info(f"Class {i} ({class_names[i]}) IoU: {class_iou * 100:.2f}%")
        
        # 记录每个文件的指标
        for fname, metrics in file_metrics.items():
            logger.info(f"\n文件: {fname}")
            logger.info(f"  Mean IoU: {metrics['mIoU'] * 100:.2f}%")
            logger.info(f"  Overall Accuracy: {metrics['OA'] * 100:.2f}%")
            logger.info(f"  F1 Score: {metrics['F1_score'] * 100:.2f}%")
            
        # 可视化全局评估结果 - 添加save_subplots=True参数和保存路径
        global_fig = visualize_results(global_metrics, class_names=class_names, 
                                      save_subplots=True, save_dir=viz_dir, prefix='global')
        global_fig.savefig(os.path.join(viz_dir, 'global_evaluation.png'), dpi=300, bbox_inches='tight')
        global_fig.savefig(os.path.join(viz_dir, 'global_evaluation.pdf'), format='pdf', bbox_inches='tight')
        plt.close(global_fig)
        
        # 创建文件级别的性能比较图
        if len(file_metrics) > 1:  # 只有当有多个文件时才创建比较图
            create_file_comparison_chart(file_metrics, viz_dir, class_names)
            
            # 保存文件比较指标到CSV
            save_file_comparison_to_csv(file_metrics, metrics_dir, class_names)

        # 处理并保存每个文件的预测结果
        for fname, data in file_predictions.items():
            # 将所有块的数据合并
            all_points = np.concatenate(data['points'], axis=0)
            all_colors = np.concatenate(data['colors'], axis=0)
            all_predictions = np.concatenate(data['predictions'], axis=0)
            all_true_labels = np.concatenate(data['true_labels'], axis=0)
            all_indices = np.concatenate(data['indices'], axis=0)
            
            # 为每个文件创建单独的可视化 - 也单独保存子图
            file_fig = visualize_results(file_metrics[fname], 
                                        class_names=class_names, 
                                        title=f"File: {fname}",
                                        save_subplots=True,
                                        save_dir=viz_dir,
                                        prefix=f'file_{fname}')
            file_fig.savefig(os.path.join(viz_dir, f'evaluation_{fname}.png'), dpi=300, bbox_inches='tight')
            file_fig.savefig(os.path.join(viz_dir, f'evaluation_{fname}.pdf'), format='pdf', bbox_inches='tight')
            plt.close(file_fig)
            
            # 可视化点云分类结果
            point_cloud_vis = visualize_point_cloud(all_points, all_predictions, all_true_labels, class_names)
            point_cloud_vis.savefig(os.path.join(viz_dir, f'pointcloud_{fname}.png'), dpi=300, bbox_inches='tight')
            point_cloud_vis.savefig(os.path.join(viz_dir, f'pointcloud_{fname}.pdf'), format='pdf', bbox_inches='tight')
            plt.close(point_cloud_vis)
                        
            # 创建新的las文件
            output_path = output_dir / f'predicted_{fname}.las'
            create_new_las_file(all_points, all_colors, all_predictions, output_path)
            logger.info(f'Saved predicted result to {output_path}')

        logger.info(f"所有可视化结果已保存至 {viz_dir}")

def save_metrics_to_csv(metrics, save_dir, filename_prefix, class_names):
    """将评估指标保存为CSV格式"""
    # 1. 保存整体性能指标
    performance_df = pd.DataFrame({
        'Metric': ['mIoU', 'OA', 'mAcc', 'Precision', 'Recall', 'F1_score'],
        'Value': [
            metrics['mIoU'] * 100,
            metrics['OA'] * 100,
            metrics['mAcc'] * 100,
            metrics['Precision'] * 100,
            metrics['Recall'] * 100,
            metrics['F1_score'] * 100
        ]
    })
    performance_df.to_csv(os.path.join(save_dir, f'{filename_prefix}_performance.csv'), index=False)
    
    # 2. 保存每个类别的IoU
    class_iou_df = pd.DataFrame({
        'Class': class_names,
        'IoU': metrics['IoU_per_class'] * 100
    })
    class_iou_df.to_csv(os.path.join(save_dir, f'{filename_prefix}_class_iou.csv'), index=False)
    
    # 3. 保存每个类别的准确率
    class_acc_df = pd.DataFrame({
        'Class': class_names,
        'Accuracy': metrics['Acc_per_class'] * 100
    })
    class_acc_df.to_csv(os.path.join(save_dir, f'{filename_prefix}_class_accuracy.csv'), index=False)
    
    # 4. 保存混淆矩阵
    cm_df = pd.DataFrame(metrics['Confusion_Matrix'], 
                         index=class_names, 
                         columns=class_names)
    cm_df.to_csv(os.path.join(save_dir, f'{filename_prefix}_confusion_matrix.csv'))
    
    # 5. 保存归一化混淆矩阵
    cm = metrics['Confusion_Matrix']
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    cm_norm_df = pd.DataFrame(cm_normalized, 
                             index=class_names, 
                             columns=class_names)
    cm_norm_df.to_csv(os.path.join(save_dir, f'{filename_prefix}_confusion_matrix_normalized.csv'))

def save_file_comparison_to_csv(file_metrics, save_dir, class_names):
    """保存文件间比较的指标为CSV"""
    # 1. 不同文件的整体性能比较
    file_names = list(file_metrics.keys())
    miou_values = [metrics['mIoU'] * 100 for metrics in file_metrics.values()]
    accuracy_values = [metrics['OA'] * 100 for metrics in file_metrics.values()]
    f1_values = [metrics['F1_score'] * 100 for metrics in file_metrics.values()]
    
    comparison_df = pd.DataFrame({
        'File': file_names,
        'mIoU': miou_values,
        'Accuracy': accuracy_values,
        'F1_Score': f1_values
    })
    comparison_df.to_csv(os.path.join(save_dir, 'file_comparison.csv'), index=False)
    
    # 2. 不同文件的类别IoU比较
    class_data = []
    for fname, metrics in file_metrics.items():
        for i, class_iou in enumerate(metrics['IoU_per_class']):
            class_data.append({
                'File': fname,
                'Class': class_names[i],
                'IoU': class_iou * 100
            })
    
    class_df = pd.DataFrame(class_data)
    class_df.to_csv(os.path.join(save_dir, 'class_iou_comparison.csv'), index=False)
    
    # 3. 创建类别IoU透视表
    pivot_df = class_df.pivot(index='Class', columns='File', values='IoU')
    pivot_df.to_csv(os.path.join(save_dir, 'class_iou_pivot.csv'))

def visualize_results(metrics, class_names=None, title=None, save_subplots=False, save_dir=None, prefix=''):
    """
    优化的评估结果可视化函数，可选择保存子图
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(metrics['IoU_per_class']))]

    # 设置科学绘图风格
    with plt.style.context('default'):
        fig = plt.figure(figsize=(20, 15))
        
        if title:
            plt.suptitle(title, fontsize=20, fontweight='bold', y=0.98)

        # 1. 混淆矩阵可视化（左上）
        ax1 = fig.add_subplot(231)
        cm = metrics['Confusion_Matrix']
        
        # 直接使用.0f格式，这对于具有整数值的浮点数也能正确显示
        sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', square=True,
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cbar_kws={'shrink': 0.7, 'label': 'Count'},
                    linewidths=0.5)
        ax1.set_title('Confusion Matrix', fontweight='bold', pad=20)
        ax1.set_xlabel('Predicted Label', fontweight='bold')
        ax1.set_ylabel('True Label', fontweight='bold')
        ax1.tick_params(axis='both', which='major', pad=8)
        for tick in ax1.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha('right')
        
        # 单独保存混淆矩阵图
        if save_subplots and save_dir:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', square=True,
                      xticklabels=class_names, yticklabels=class_names,
                      cbar_kws={'shrink': 0.7, 'label': 'Count'}, linewidths=0.5)
            plt.title('Confusion Matrix', fontweight='bold', pad=20)
            plt.xlabel('Predicted Label', fontweight='bold')
            plt.ylabel('True Label', fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{prefix}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(save_dir, f'{prefix}_confusion_matrix.pdf'), format='pdf', bbox_inches='tight')
            plt.close()

        # 2. 每个类别的IoU柱状图（右上）
        ax2 = fig.add_subplot(232)
        iou_data = metrics['IoU_per_class'] * 100
        
        # 使用科学配色
        bars = ax2.bar(class_names, iou_data, color=sci_colors[:len(class_names)])
        ax2.set_title(f'IoU per Class (mIoU: {metrics["mIoU"] * 100:.2f}%)', fontweight='bold')
        ax2.set_ylabel('IoU (%)', fontweight='bold')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.set_ylim(0, 100)
        # 在每个柱子上方添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        ax2.tick_params(axis='x', rotation=45, labelsize=12)
        
        # 单独保存IoU柱状图
        if save_subplots and save_dir:
            plt.figure(figsize=(10, 8))
            bars = plt.bar(class_names, iou_data, color=sci_colors[:len(class_names)])
            plt.title(f'IoU per Class (mIoU: {metrics["mIoU"] * 100:.2f}%)', fontweight='bold')
            plt.ylabel('IoU (%)', fontweight='bold')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.ylim(0, 100)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{prefix}_iou_per_class.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(save_dir, f'{prefix}_iou_per_class.pdf'), format='pdf', bbox_inches='tight')
            plt.close()
        
        # 3. 每个类别的准确率柱状图（中上）
        ax3 = fig.add_subplot(233)
        acc_data = metrics['Acc_per_class'] * 100
        
        bars = ax3.bar(class_names, acc_data, color=sci_colors[:len(class_names)])
        ax3.set_title(f'Accuracy per Class (mAcc: {metrics["mAcc"] * 100:.2f}%)', fontweight='bold')
        ax3.set_ylabel('Accuracy (%)', fontweight='bold')
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        ax3.set_ylim(0, 100)
        # 在每个柱子上方添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        ax3.tick_params(axis='x', rotation=45, labelsize=12)

        # 单独保存准确率柱状图
        if save_subplots and save_dir:
            plt.figure(figsize=(10, 8))
            acc_data = metrics['Acc_per_class'] * 100
            bars = plt.bar(class_names, acc_data, color=sci_colors[:len(class_names)])
            plt.title(f'Accuracy per Class (mAcc: {metrics["mAcc"] * 100:.2f}%)', fontweight='bold')
            plt.ylabel('Accuracy (%)', fontweight='bold')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.ylim(0, 100)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{prefix}_accuracy_per_class.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(save_dir, f'{prefix}_accuracy_per_class.pdf'), format='pdf', bbox_inches='tight')
            plt.close()

        # 4. 归一化的混淆矩阵（左下）
        ax4 = fig.add_subplot(234)
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
        
        # 对于归一化后的矩阵，格式总是百分比
        sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='RdYlBu_r', square=True,
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cbar_kws={'shrink': 0.7, 'label': 'Percentage'},
                    linewidths=0.5)
        ax4.set_title('Normalized Confusion Matrix', fontweight='bold', pad=20)
        ax4.set_xlabel('Predicted Label', fontweight='bold')
        ax4.set_ylabel('True Label', fontweight='bold')
        ax4.tick_params(axis='both', which='major', pad=8)
        for tick in ax4.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha('right')
        
        # 单独保存归一化混淆矩阵
        if save_subplots and save_dir:
            plt.figure(figsize=(10, 8))
            cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
            sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='RdYlBu_r', square=True,
                      xticklabels=class_names, yticklabels=class_names,
                      cbar_kws={'shrink': 0.7, 'label': 'Percentage'}, linewidths=0.5)
            plt.title('Normalized Confusion Matrix', fontweight='bold', pad=20)
            plt.xlabel('Predicted Label', fontweight='bold')
            plt.ylabel('True Label', fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{prefix}_confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(save_dir, f'{prefix}_confusion_matrix_normalized.pdf'), format='pdf', bbox_inches='tight')
            plt.close()

        # 5. Precision, Recall, F1 Score柱状图（右下）
        ax5 = fig.add_subplot(235)
        
        prec_recall_f1 = [
            metrics['Precision'] * 100,
            metrics['Recall'] * 100,
            metrics['F1_score'] * 100,
            metrics['OA'] * 100
        ]
        
        metrics_names = ['Precision', 'Recall', 'F1 Score', 'Overall Acc']
        bars = ax5.bar(metrics_names, prec_recall_f1, color=sci_colors[5:9])
        ax5.set_title('Overall Performance Metrics', fontweight='bold')
        ax5.set_ylabel('Percentage (%)', fontweight='bold')
        ax5.grid(axis='y', linestyle='--', alpha=0.7)
        ax5.set_ylim(0, 100)
        # 在每个柱子上方添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')

        # 单独保存性能指标柱状图
        if save_subplots and save_dir:
            plt.figure(figsize=(10, 8))
            prec_recall_f1 = [
                metrics['Precision'] * 100,
                metrics['Recall'] * 100,
                metrics['F1_score'] * 100,
                metrics['OA'] * 100
            ]
            metrics_names = ['Precision', 'Recall', 'F1 Score', 'Overall Acc']
            bars = plt.bar(metrics_names, prec_recall_f1, color=sci_colors[5:9])
            plt.title('Overall Performance Metrics', fontweight='bold')
            plt.ylabel('Percentage (%)', fontweight='bold')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.ylim(0, 100)
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{prefix}_performance_metrics.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(save_dir, f'{prefix}_performance_metrics.pdf'), format='pdf', bbox_inches='tight')
            plt.close()

        # 6. 类别样本分布饼图（中下）
        ax6 = fig.add_subplot(236)
        class_counts = np.sum(cm, axis=1)
        
        # 添加百分比到标签中
        total = class_counts.sum()
        class_percents = [f'{name} ({count/total*100:.1f}%)' for name, count in zip(class_names, class_counts)]
        
        wedges, texts, autotexts = ax6.pie(
            class_counts, 
            labels=class_percents,
            autopct='%1.1f%%', 
            startangle=90, 
            colors=sci_colors[:len(class_names)],
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}
        )
        
        # 设置饼图中文本的样式
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
            
        for text in texts:
            text.set_fontsize(10)
        
        ax6.set_title('Class Distribution', fontweight='bold')
        ax6.axis('equal')  # 确保饼图是圆的

        # 单独保存类别分布饼图
        if save_subplots and save_dir:
            plt.figure(figsize=(10, 8))
            class_counts = np.sum(cm, axis=1)
            total = class_counts.sum()
            class_percents = [f'{name} ({count/total*100:.1f}%)' for name, count in zip(class_names, class_counts)]
            wedges, texts, autotexts = plt.pie(
                class_counts, 
                labels=class_percents,
                autopct='%1.1f%%', 
                startangle=90, 
                colors=sci_colors[:len(class_names)],
                wedgeprops={'edgecolor': 'w', 'linewidth': 1}
            )
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_fontweight('bold')
            for text in texts:
                text.set_fontsize(10)
            plt.title('Class Distribution', fontweight='bold')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{prefix}_class_distribution.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(save_dir, f'{prefix}_class_distribution.pdf'), format='pdf', bbox_inches='tight')
            plt.close()

        # 其余代码保持不变
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为标题留出空间
        return fig

def visualize_point_cloud(points, predictions, true_labels, class_names):
    """
    可视化点云分类结果
    """
    fig = plt.figure(figsize=(18, 10))

    # 创建自定义颜色映射（科学风格）
    colors = ['#4878d0', '#ee854a', '#6acc64', '#d65f5f', '#956cb4', '#8c613c', '#dc7ec0', '#797979', '#d5bb67', '#82c6e2']
    class_colors = colors[:len(class_names)]
    
    # 真实标签点云
    ax1 = fig.add_subplot(121, projection='3d')
    
    # 获取唯一的标签
    unique_labels = np.unique(true_labels)
    
    # 为每个类别创建散点图
    for i, label in enumerate(unique_labels):
        mask = true_labels == label
        if np.sum(mask) > 0:  # 只绘制有点的类别
            ax1.scatter(
                points[mask, 0], points[mask, 1], points[mask, 2],
                s=2, 
                c=[class_colors[label]],
                label=class_names[label]
            )
    ax1.set_title('Ground Truth', fontsize=16, fontweight='bold')
    ax1.set_xlabel('X', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Y', fontsize=14, fontweight='bold')
    ax1.set_zlabel('Z', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True, fontsize=12)
    
    # 预测标签点云
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 获取唯一的预测标签
    unique_predictions = np.unique(predictions)
    
    # 为每个预测类别创建散点图
    for i, label in enumerate(unique_predictions):
        mask = predictions == label
        if np.sum(mask) > 0:  # 只绘制有点的类别
            ax2.scatter(
                points[mask, 0], points[mask, 1], points[mask, 2],
                s=2, 
                c=[class_colors[label]],
                label=class_names[label]
            )
    ax2.set_title('Prediction', fontsize=16, fontweight='bold')
    ax2.set_xlabel('X', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Y', fontsize=14, fontweight='bold')
    ax2.set_zlabel('Z', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, fontsize=12)

    # 设置相同的视角
    for ax in [ax1, ax2]:
        ax.view_init(elev=30, azim=45)
        ax.set_box_aspect([1, 1, 0.5])  # 设置纵横比

    plt.tight_layout()
    return fig

def create_file_comparison_chart(file_metrics, viz_dir, class_names):
    """
    创建多文件性能比较图
    """
    plt.figure(figsize=(15, 10))
    
    # 提取每个文件的IoU
    file_names = list(file_metrics.keys())
    miou_values = [metrics['mIoU'] * 100 for metrics in file_metrics.values()]
    accuracy_values = [metrics['OA'] * 100 for metrics in file_metrics.values()]
    f1_values = [metrics['F1_score'] * 100 for metrics in file_metrics.values()]
    
    # 创建DataFrame以便于绘图
    df = pd.DataFrame({
        'File': file_names * 3,
        'Metric': ['mIoU'] * len(file_names) + ['Accuracy'] * len(file_names) + ['F1 Score'] * len(file_names),
        'Value': miou_values + accuracy_values + f1_values
    })
    
    # 创建分组柱状图
    plt.figure(figsize=(12, 8))
    g = sns.barplot(x='File', y='Value', hue='Metric', data=df, palette=sci_colors[:3])
    
    # 添加数值标签
    for container in g.containers:
        g.bar_label(container, fmt='%.1f%%', fontsize=10)
    
    plt.title('Performance Metrics by File', fontsize=18, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Metric', title_fontsize=12, fontsize=12, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'file_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(viz_dir, 'file_comparison.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

    # 为每个类别的IoU创建比较图
    plt.figure(figsize=(15, 10))
    
    # 构建DataFrame
    class_data = []
    for fname, metrics in file_metrics.items():
        for i, class_iou in enumerate(metrics['IoU_per_class']):
            class_data.append({
                'File': fname,
                'Class': class_names[i],
                'IoU': class_iou * 100
            })
    
    class_df = pd.DataFrame(class_data)
    
    # 创建热力图
    plt.figure(figsize=(12, 8))
    heatmap_data = class_df.pivot(index='Class', columns='File', values='IoU')
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlGnBu',
                linewidths=0.5, cbar_kws={'label': 'IoU (%)'})
    plt.title('Class IoU by File (%)', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'class_iou_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(viz_dir, 'class_iou_comparison.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

def create_new_las_file(points, colors, labels, output_path):
    """创建新的las文件，只包含xyz, rgb和classification"""
    # 创建LAS文件
    las = laspy.create(file_version="1.3", point_format=3)

    # 写入点数据
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    # 设置头部信息
    #las.header.offsets = [np.min(las.x), np.min(las.y), np.min(las.z)]
    #las.header.scales = [0.001, 0.001, 0.001]
    
    # 设置RGB值 (需要从0-1范围转换到0-65535范围)
    las.red = (colors[:, 0] * 65535).astype(np.uint16)
    las.green = (colors[:, 1] * 65535).astype(np.uint16)
    las.blue = (colors[:, 2] * 65535).astype(np.uint16)
    
    # 设置分类标签
    las.classification = labels.astype(np.uint8)
    
    # 写入文件
    las.write(output_path)


# 定义评价指标计算函数
def calculate_metrics(confusion_matrix):
    """
    从混淆矩阵计算所有指标
    """
    cm = confusion_matrix.cpu().numpy()

    # 计算每个类别的IoU
    intersection = np.diag(cm)
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)
    iou_per_class = intersection / (union + 1e-6)
    miou = np.nanmean(iou_per_class)

    # 计算OA (Overall Accuracy)
    oa = np.sum(np.diag(cm)) / np.sum(cm)

    # 计算每个类别的准确率
    acc_per_class = np.diag(cm) / (np.sum(cm, axis=1) + 1e-6)
    macc = np.nanmean(acc_per_class)  # Mean Accuracy

    # 计算每个类的precision和recall
    precision_per_class = np.diag(cm) / (np.sum(cm, axis=0) + 1e-6)
    recall_per_class = np.diag(cm) / (np.sum(cm, axis=1) + 1e-6)

    # 计算加权平均的precision和recall
    weights = np.sum(cm, axis=1) / np.sum(cm)
    precision = np.sum(precision_per_class * weights)
    recall = np.sum(recall_per_class * weights)

    # 计算F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return {
        'mIoU': miou,
        'IoU_per_class': iou_per_class,
        'OA': oa,
        'mAcc': macc,
        'Acc_per_class': acc_per_class,
        'Precision': precision,
        'Recall': recall,
        'F1_score': f1,
        'Confusion_Matrix': cm
    }



class AverageMeter(object):
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def test():
    # 设置随机种子保证可复现性
    torch.manual_seed(42)

    # 创建测试数据
    batch_size = 2
    num_points = 1024
    num_classes = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建随机输入数据
    xyz = torch.randn(batch_size, num_points, 3)
    features = torch.randn(batch_size, num_points, 3)

    # from models.enhanced_pointnet2 import EnhancedPointNet2
    from experiments.exp_010619_brimulti_brienc_CB_all_weightedloss_bsize1.models.enhanced_pointnet2 import EnhancedPointNet2
    model = EnhancedPointNet2(num_classes)

    checkpoint_path = 'experiments/exp_010619_brimulti_brienc_CB_all_weightedloss_bsize1/latest_checkpoint.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded model from {checkpoint_path}')
    else:
        print(f'No checkpoint found at {checkpoint_path}')
        return

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


if __name__ == '__main__':
    main()
    #test()
