import os

import laspy
import numpy as np
import torch
from torch.utils.data import Dataset

from .logger_config import get_logger

logger = get_logger()

#分块，重叠，采样点数，数据增强


class BridgePointCloudDataset(Dataset):
    def __init__(self, data_dir, num_points=4096, transform=False, chunk_size=8192, overlap=1024):
        super().__init__()
        self.data_dir = data_dir
        self.num_points = num_points
        self.transform = transform
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.valid_labels = set()

        # 检查目录
        if not os.path.exists(data_dir):
            raise ValueError(f"数据目录不存在: {data_dir}")

        # 获取文件列表
        self.file_list = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith('.las')
        ]

        if len(self.file_list) == 0:
            raise ValueError(f"在目录 {data_dir} 中没有找到.las文件")

        # 预加载所有数据
        logger.info("开始预加载数据到内存...")
        self.cached_chunks = []

        for file_idx, las_path in enumerate(self.file_list):
            las = laspy.read(las_path)
            num_points = len(las.points)

            # 提取所有点的数据
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

            # 正规化点云
            points = self.normalize_points(points)

            # 分块并缓存
            num_chunks = max(1, (num_points - overlap) // (chunk_size - overlap))
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * (chunk_size - overlap)
                end_idx = min(start_idx + chunk_size, num_points)

                chunk_points = torch.from_numpy(points[start_idx:end_idx].astype(np.float32))
                chunk_colors = torch.from_numpy(colors[start_idx:end_idx].astype(np.float32))
                chunk_labels = torch.from_numpy(labels[start_idx:end_idx].astype(np.int64))

                # 使用FPS进行采样
                if len(chunk_points) > self.num_points:
                    chunk_points = chunk_points.unsqueeze(0)
                    chunk_colors = chunk_colors.unsqueeze(0)
                    chunk_labels = chunk_labels.unsqueeze(0)

                    sampled_indices = farthest_point_sample(chunk_points, self.num_points)
                    chunk_points = index_points(chunk_points, sampled_indices).squeeze(0)
                    chunk_colors = index_points(chunk_colors, sampled_indices).squeeze(0)
                    chunk_labels = index_points(chunk_labels.unsqueeze(-1), sampled_indices).squeeze(0)

                self.cached_chunks.append({
                    'points': chunk_points,
                    'colors': chunk_colors,
                    'labels': chunk_labels
                })

        logger.info(f"数据预加载完成，共 {len(self.cached_chunks)} 个数据块")
        logger.info(f"有效标签: {sorted(list(self.valid_labels))}")

    def __len__(self):
        return len(self.cached_chunks)

    def __getitem__(self, idx):
        data = self.cached_chunks[idx]
        points = data['points']
        colors = data['colors']
        labels = data['labels']

        if self.transform:
            points, colors = self.apply_transform(points.numpy(), colors.numpy())
            points = torch.from_numpy(points.astype(np.float32))
            colors = torch.from_numpy(colors.astype(np.float32))

        return {
            'points': points,
            'colors': colors,
            'labels': labels
        }


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
        super().__init__(data_dir, num_points, transform=False, chunk_size=chunk_size, overlap=overlap)

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
