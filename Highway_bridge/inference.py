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
    model.eval()
    total_points = len(points)
    predictions = np.zeros(total_points, dtype=np.int64)
    
    # 预处理数据
    points, colors = preprocess_for_inference(points, colors)

    # 批处理大小
    batch_size = 32  # 可以根据GPU显存调整这个值
    
    # 准备数据批次
    batch_data = []
    batch_indices = []
    current_batch = []
    current_indices = []
    
    # 使用tqdm显示处理进度
    for i in tqdm(range(0, total_points, num_points), desc="Preparing batches"):
        end_idx = min(i + num_points, total_points)
        current_points = points[i:end_idx]
        current_colors = colors[i:end_idx]
        
        # 如果不足num_points，进行填充
        if len(current_points) < num_points:
            pad_size = num_points - len(current_points)
            current_points = np.pad(current_points, ((0, pad_size), (0, 0)), mode='wrap')
            current_colors = np.pad(current_colors, ((0, pad_size), (0, 0)), mode='wrap')
        
        current_batch.append((current_points, current_colors))
        current_indices.append((i, end_idx))
        
        # 当收集够一个批次或是最后一批时，进行处理
        if len(current_batch) == batch_size or end_idx == total_points:
            batch_points = np.stack([b[0] for b in current_batch])
            batch_colors = np.stack([b[1] for b in current_batch])
            
            # 转换为tensor并移到GPU
            batch_points = torch.FloatTensor(batch_points).to(device)
            batch_colors = torch.FloatTensor(batch_colors).to(device)
            
            # 预测
            with torch.no_grad():
                outputs = model(batch_points, batch_colors)
                preds = outputs.max(1)[1].cpu().numpy()
            
            # 将预测结果存储到对应位置
            for (start_idx, end_idx), pred in zip(current_indices, preds):
                actual_size = min(end_idx - start_idx, num_points)
                predictions[start_idx:end_idx] = pred[:actual_size]
            
            # 清空当前批次
            current_batch = []
            current_indices = []
    
    return predictions




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

def main():
    num_classes = 5
    logger = setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 加载模型
    model = EnhancedPointNet2(num_classes).to(device)
    checkpoint_path = 'experiments/exp_lindata/best_model.pth'
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'Loaded model from {checkpoint_path}')
    else:
        logger.error(f'No checkpoint found at {checkpoint_path}')
        return
    
    model.eval()
    
    # 创建输出目录
    output_dir = Path('data/predicted_las')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理测试文件夹中的所有.las文件
    test_dir = Path('data/test')
    test_files = list(test_dir.glob('*.las'))
    
    for file_path in tqdm(test_files, desc='Processing files'):
        try:
            logger.info(f'Processing {file_path}')
            # 读取las文件
            las_in = laspy.read(file_path)
            
            # 提取xyz坐标
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
            #predictions = labels = np.array(las_in.classification)

            # 确保预测结果长度与点云数量一致
            assert len(predictions) == len(points), f"Prediction length {len(predictions)} doesn't match points length {len(points)}"

            # 保存为新的las文件
            output_path = output_dir / f'predicted_{file_path.name}'
            create_new_las_file(points, colors, predictions, str(output_path))
            
            logger.info(f'Successfully processed and saved: {output_path}')
            
        except Exception as e:
            logger.error(f'Error processing {file_path}: {str(e)}')
            continue

if __name__ == '__main__':
    main()
