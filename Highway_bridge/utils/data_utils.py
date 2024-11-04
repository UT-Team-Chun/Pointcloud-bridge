import os
import numpy as np
import laspy
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


class BridgePointCloudDataset(Dataset):
    def __init__(self, data_dir, num_points=4096, transform=False, chunk_size=8192, overlap=1024):
        """
        Args:
            data_dir (str): 包含.las文件的目录路径
            num_points (int): 最终采样点数
            transform (bool): 是否进行数据增强
            chunk_size (int): 每个块的初始点数
            overlap (int): 块之间的重叠点数
        """
        self.data_dir = data_dir
        self.num_points = num_points  # 保持最终输出点数不变
        self.transform = transform
        self.chunk_size = chunk_size
        self.overlap = overlap

        # 检查目录是否存在
        if not os.path.exists(data_dir):
            raise ValueError(f"数据目录不存在: {data_dir}")

        # 获取所有.las文件路径
        self.file_list = []
        for file in os.listdir(data_dir):
            if file.endswith('.las'):
                self.file_list.append(os.path.join(data_dir, file))

        if len(self.file_list) == 0:
            raise ValueError(f"在目录 {data_dir} 中没有找到.las文件")

        # 添加标签验证
        self.valid_labels = set()  # 将在validate_dataset中填充
        self.validate_dataset()

        # 预计算每个文件的块数和映射关系
        self.chunk_to_file_map = []
        for file_idx, las_path in enumerate(self.file_list):
            las = laspy.read(las_path)
            num_points = len(las.points)
            # 计算当前文件需要多少个chunk
            num_chunks = max(1, (num_points - overlap) // (chunk_size - overlap))
            # 记录每个chunk对应的文件信息
            for chunk_idx in range(num_chunks):
                self.chunk_to_file_map.append((file_idx, chunk_idx))

        logger.info(f"在 {data_dir} 中找到 {len(self.file_list)} 个.las文件")
        logger.info(f"总块数: {len(self.chunk_to_file_map)}")
        logger.info(f"数据集中的有效标签: {sorted(list(self.valid_labels))}")

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

    def __len__(self):
        return len(self.chunk_to_file_map)

    def __getitem__(self, idx):
        # 获取当前chunk对应的文件信息
        file_idx, chunk_idx = self.chunk_to_file_map[idx]
        las_path = self.file_list[file_idx]

        try:
            las = laspy.read(las_path)

            # 计算当前chunk的起始和结束索引
            start_idx = chunk_idx * (self.chunk_size - self.overlap)
            end_idx = min(start_idx + self.chunk_size, len(las.points))

            # 提取点云数据
            points = np.vstack((
                las.x[start_idx:end_idx],
                las.y[start_idx:end_idx],
                las.z[start_idx:end_idx]
            )).T

            # 提取颜色信息
            if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                colors = np.vstack((
                    np.array(las.red[start_idx:end_idx]) / 65535.0,
                    np.array(las.green[start_idx:end_idx]) / 65535.0,
                    np.array(las.blue[start_idx:end_idx]) / 65535.0
                )).T
            else:
                colors = np.zeros((points.shape[0], 3))

            # 提取分类标签
            if hasattr(las, 'classification'):
                labels = np.array(las.classification[start_idx:end_idx])
                labels = np.where((labels >= 0) & (labels <= 7), labels, 0)
            else:
                labels = np.zeros(points.shape[0])

            # 从chunk中随机采样到指定点数
            if len(points) > self.num_points:
                indices = np.random.choice(len(points), self.num_points, replace=False)
            else:
                indices = np.random.choice(len(points), self.num_points, replace=True)

            points = points[indices]
            colors = colors[indices]
            labels = labels[indices]

            # 应用正规化
            points = self.normalize_points(points)
            colors = self.normalize_colors(colors)

            # 数据增强
            if self.transform:
                points, colors = self.apply_transform(points, colors)

            # 转换为张量
            points = torch.from_numpy(points.astype(np.float32))
            colors = torch.from_numpy(colors.astype(np.float32))
            labels = torch.from_numpy(labels.astype(np.int64))

            return {
                'points': points,
                'colors': colors,
                'labels': labels
            }

        except Exception as e:
            logger.error(f"处理文件 {las_path} 时出错: {str(e)}")
            raise

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

