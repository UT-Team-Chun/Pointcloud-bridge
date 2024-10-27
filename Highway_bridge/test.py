import torch
import numpy as np
import laspy
from pathlib import Path
from tqdm import tqdm
import logging
from models.enhanced_pointnet2 import EnhancedPointNet2
import os

def preprocess_for_inference(points, colors):
    """
    推理时使用的预处理函数
    
    Args:
        points: numpy array of shape (N, 3)
        colors: numpy array of shape (N, 3)
    
    Returns:
        points: normalized points
        colors: normalized colors
    """
    # 正规化点云坐标
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    if max_dist > 0:
        points = points / max_dist
    
    # 确保颜色值在0-1范围内
    colors = np.clip(colors, 0, 1)
    
    return points, colors

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('testing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def process_point_cloud(points, colors, model, device, num_points=4096):
    """处理单个点云文件"""
    total_points = len(points)
    predictions = np.zeros(total_points, dtype=np.int64)
    
    # 预处理数据
    points, colors = preprocess_for_inference(points, colors)

    for i in range(0, total_points, num_points):
        end_idx = min(i + num_points, total_points)
        batch_points = points[i:end_idx]
        batch_colors = colors[i:end_idx]
        
        # 如果批次大小不足，需要补齐
        if len(batch_points) < num_points:
            pad_size = num_points - len(batch_points)
            batch_points = np.concatenate([batch_points, batch_points[:pad_size]], axis=0)
            batch_colors = np.concatenate([batch_colors, batch_colors[:pad_size]], axis=0)
        
        # 转换为tensor
        batch_points = torch.FloatTensor(batch_points).unsqueeze(0).to(device)
        batch_colors = torch.FloatTensor(batch_colors).unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(batch_points, batch_colors)
            pred = outputs.max(1)[1].cpu().numpy()
        
        # 只保存实际点的预测结果
        predictions[i:end_idx] = pred[:end_idx-i]
    
    return predictions

def create_new_las_file(points, colors, labels, output_path):
    """创建新的las文件，只包含xyz, rgb和classification"""

    # 创建LAS文件
    las = laspy.create(file_version="1.3", point_format=3)

    # 写入点数据
    # 设置xyz坐标
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    # 设置头部信息
    las.header.offsets = [np.min(las.x), np.min(las.y), np.min(las.z)]
    las.header.scales = [0.001, 0.001, 0.001]
    
    # 设置RGB值 (需要从0-1范围转换到0-65535范围)
    las.red = (colors[:, 0] * 65535).astype(np.uint16)
    las.green = (colors[:, 1] * 65535).astype(np.uint16)
    las.blue = (colors[:, 2] * 65535).astype(np.uint16)
    
    # 设置分类标签
    las.classification = labels
    
    # 写入文件
    las.write(output_path)

def main():
    logger = setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 加载模型
    model = EnhancedPointNet2(num_classes=8).to(device)
    checkpoint_path = 'experiments/exp_20241027_193536/best_model.pth'  # 替换为你的模型路径
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'Loaded model from {checkpoint_path}')
    else:
        logger.error(f'No checkpoint found at {checkpoint_path}')
        return
    
    model.eval()
    
    # 创建输出目录
    output_dir = Path('results/predicted_las')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理测试文件夹中的所有.las文件
    test_dir = Path('data/test')
    test_files = list(test_dir.glob('*.las'))
    
    for file_path in tqdm(test_files, desc='Processing files'):
        try:
            # 读取las文件
            las_in = laspy.read(file_path)
            
            # 只提取xyz坐标
            points = np.vstack((las_in.x, las_in.y, las_in.z)).T
            
            # 提取并归一化RGB值
            if hasattr(las_in, 'red'):
                colors = np.vstack((
                    las_in.red / 65535.0,
                    las_in.green / 65535.0,
                    las_in.blue / 65535.0
                )).T
            else:
                colors = np.ones((points.shape[0], 3)) * 0.5
            
            # 获取预测结果
            predictions = process_point_cloud(points, colors, model, device)

            # 保存为新的las文件
            output_path = output_dir / f'predicted_{file_path.name}'
            output_path = str(output_path)

            create_new_las_file(points, colors, predictions, output_path)
            
            logger.info(f'Successfully processed and saved: {output_path}')
            
        except Exception as e:
            logger.error(f'Error processing {file_path}: {str(e)}')
            continue

if __name__ == '__main__':
    main()
