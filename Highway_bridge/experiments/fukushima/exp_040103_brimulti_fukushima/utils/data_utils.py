import os

import laspy
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .logger_config import get_logger

logger = get_logger()

#分块，重叠，采样点数，数据增强


class BridgePointCloudDataset(Dataset):
    def __init__(self, data_dir, num_points=4096, block_size=1.0, overlap=0.3):
        super().__init__()
        self.data_dir = data_dir
        self.num_points = num_points
        self.block_size = block_size
        self.overlap = overlap
        self.valid_labels = set()
        self.cached_chunks = []

        if not os.path.exists(data_dir):
            raise ValueError(f"数据目录不存在: {data_dir}")

        self.file_list = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith('.las')
        ]

        if len(self.file_list) == 0:
            raise ValueError(f"在目录 {data_dir} 中没有找到.las文件")

        logger.info("开始预加载数据到内存...")

        for file_idx, las_path in enumerate(tqdm(self.file_list, desc="loading data")):
            # 读取点云数据
            las = laspy.read(las_path)
            points = np.vstack((las.x, las.y, las.z)).T

            # 提取颜色
            if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                colors = np.vstack((
                    np.array(las.red) / 65535.0,
                    np.array(las.green) / 65535.0,
                    np.array(las.blue) / 65535.0
                )).T
            else:
                colors = np.zeros((points.shape[0], 3))

            # 提取标签
            if hasattr(las, 'classification'):
                labels = np.array(las.classification)
                labels = np.where((labels >= 0) & (labels <= 7), labels, 0)
                self.valid_labels.update(np.unique(labels))
            else:
                labels = np.zeros(points.shape[0])

            # 计算边界框
            min_bound = np.min(points, axis=0)
            max_bound = np.max(points, axis=0)

            # 计算网格步长（考虑重叠）
            step = self.block_size * (1 - self.overlap)

            # 使用网格划分方法
            x_steps = int((max_bound[0] - min_bound[0]) / step) + 1
            y_steps = int((max_bound[1] - min_bound[1]) / step) + 1

            # 创建网格中心点
            for i in range(x_steps):
                for j in range(y_steps):
                    center_x = min_bound[0] + i * step + self.block_size / 2
                    center_y = min_bound[1] + j * step + self.block_size / 2

                    # 找到block范围内的点
                    mask = ((points[:, 0] >= center_x - self.block_size / 2) &
                            (points[:, 0] < center_x + self.block_size / 2) &
                            (points[:, 1] >= center_y - self.block_size / 2) &
                            (points[:, 1] < center_y + self.block_size / 2))

                    block_points = points[mask]
                    block_colors = colors[mask]
                    block_labels = labels[mask]

                    if len(block_points) < 100:  # 跳过点数太少的块
                        continue

                    # 采样或填充到指定点数
                    if len(block_points) > self.num_points:
                        # 随机采样
                        indices = np.random.choice(len(block_points), self.num_points, replace=False)
                    else:
                        # 重复采样
                        indices = np.random.choice(len(block_points), self.num_points, replace=True)

                    block_points = block_points[indices]
                    block_colors = block_colors[indices]
                    block_labels = block_labels[indices]

                    # 中心化
                    block_points = block_points - np.array([center_x, center_y, np.mean(block_points[:, 2])])

                    # 转换为tensor
                    block_points = torch.from_numpy(block_points.astype(np.float32))
                    block_colors = torch.from_numpy(block_colors.astype(np.float32))
                    block_labels = torch.from_numpy(block_labels.astype(np.int64))

                    self.cached_chunks.append({
                        'points': block_points,
                        'colors': block_colors,
                        'labels': block_labels
                    })

    def __len__(self):
        return len(self.cached_chunks)

    def __getitem__(self, idx):
        return self.cached_chunks[idx]

    def normalize_points(self, points):
        """正规化点云坐标"""
        # 计算质心
        centroid = np.mean(points, axis=0)
        points = points - centroid

        # 计算最大距离
        max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if max_dist > 0:
            points = points / max_dist

        return points

    def normalize_colors(self, colors):
        """确保颜色值在0-1范围内"""
        return np.clip(colors, 0, 1)

    def validate_dataset(self):
        """验证整个数据集，收集所有可能的标签"""
        logger.info("开始验证数据集...")
        for las_path in self.file_list:
            try:
                las = laspy.read(las_path)
                if hasattr(las, 'classification'):
                    unique_labels = np.unique(las.classification)
                    self.valid_labels.update(unique_labels)
            except Exception as e:
                logger.error(f"验证文件 {las_path} 时出错: {str(e)}")
                raise
        logger.info("数据集验证完成")

    def apply_transform(self, points, colors):
        """数据增强函数"""
        if not self.transform:
            return points, colors

        # 随机旋转
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        points = np.dot(points, rotation_matrix)

        # 随机平移
        translation = np.random.uniform(-0.2, 0.2, size=(1, 3))
        points += translation

        # 随机缩放
        scale = np.random.uniform(0.8, 1.2)
        points *= scale

        # 随机抖动颜色
        if colors is not None:
            color_noise = np.random.normal(0, 0.02, colors.shape)
            colors = np.clip(colors + color_noise, 0, 1)

        return points, colors

class BridgeValidationDataset(BridgePointCloudDataset):
    def __init__(self, data_dir, num_points=4096, chunk_size=8192, overlap=1024, validation_ratio=0.3, seed=42):
        # 调用父类的__init__，但强制transform=False
        super().__init__(data_dir, num_points, overlap=overlap)

        # 设置随机种子
        np.random.seed(seed)

        # 随机采样30%的数据块
        total_chunks = len(self.cached_chunks)
        num_val_chunks = int(total_chunks * validation_ratio)

        # 随机选择索引
        selected_indices = np.random.choice(total_chunks, num_val_chunks, replace=False)

        # 只保留选中的数据块
        self.cached_chunks = [self.cached_chunks[i] for i in selected_indices]

        logger.info(
            f"验证集创建完成，使用 {len(self.cached_chunks)} 个数据块 (总共 {total_chunks} 个的 {validation_ratio * 100:.1f}%)")
        logger.info(f"有效标签: {sorted(list(self.valid_labels))}")

    def __getitem__(self, idx):
        # 简化的getitem，移除了数据增强
        data = self.cached_chunks[idx]
        return {
            'points': data['points'],
            'colors': data['colors'],
            'labels': data['labels']
        }


def farthest_point_sample(xyz, npoint):
    """最远点采样"""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """球查询"""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx

def square_distance(src, dst):
    """计算两组点之间的欧氏距离"""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """根据索引从点云中获取对应的点"""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
