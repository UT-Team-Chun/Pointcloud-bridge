import os
import laspy
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.neighbors import KDTree

from .logger_config import get_logger

logger = get_logger()


class BridgePointCloudDataset(Dataset):
    def __init__(self, data_dir, num_points=4096, block_size=1.0, overlap=0.3, min_points=100):
        super().__init__()
        self.data_dir = data_dir
        self.num_points = num_points
        self.block_size = block_size
        self.overlap = overlap
        self.min_points = min_points
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
        self.load_and_process_data()

    def load_and_process_data(self):
        """加载并处理所有数据文件"""
        for file_idx, las_path in enumerate(tqdm(self.file_list, desc="Loading data")):
            try:
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

                # 3D分块处理
                chunks = self.create_3d_blocks(points, colors, labels)
                self.cached_chunks.extend(chunks)

            except Exception as e:
                logger.error(f"处理文件 {las_path} 时出错: {str(e)}")
                raise

        logger.info(f"数据加载完成，共创建 {len(self.cached_chunks)} 个数据块")

    def create_3d_blocks(self, points, colors, labels):
        """创建3D数据块"""
        # 计算3D边界框
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)

        # 计算步长（考虑重叠）
        step = self.block_size * (1 - self.overlap)

        # 计算每个维度的网格数
        x_steps = max(int((max_bound[0] - min_bound[0]) / step) + 1, 1)
        y_steps = max(int((max_bound[1] - min_bound[1]) / step) + 1, 1)
        z_steps = max(int((max_bound[2] - min_bound[2]) / step) + 1, 1)

        chunks = []

        # 3D网格遍历
        for i in range(x_steps):
            for j in range(y_steps):
                for k in range(z_steps):
                    # 计算当前块的中心点
                    center = np.array([
                        min_bound[0] + i * step + self.block_size / 2,
                        min_bound[1] + j * step + self.block_size / 2,
                        min_bound[2] + k * step + self.block_size / 2
                    ])

                    # 选择在当前块范围内的点
                    mask = np.all(
                        (points >= (center - self.block_size / 2)) &
                        (points < (center + self.block_size / 2)),
                        axis=1
                    )

                    block_points = points[mask]
                    block_colors = colors[mask]
                    block_labels = labels[mask]

                    # 跳过点数太少的块
                    if len(block_points) < self.min_points:
                        continue

                    # 处理采样
                    chunk = self.process_block(block_points, block_colors, block_labels, center)
                    if chunk is not None:
                        chunks.append(chunk)

        return chunks

    def process_block(self, points, colors, labels, center):
        """处理单个数据块"""
        # 采样或填充到指定点数
        if len(points) > self.num_points:
            # 使用FPS进行采样
            indices = self.farthest_point_sampling(points)
        else:
            # 重复采样
            indices = np.random.choice(len(points), self.num_points, replace=True)

        # 提取采样后的数据
        sampled_points = points[indices]
        sampled_colors = colors[indices]
        sampled_labels = labels[indices]

        # 中心化
        centered_points = sampled_points - center

        # 转换为tensor
        return {
            'points': torch.from_numpy(centered_points.astype(np.float32)),
            'colors': torch.from_numpy(sampled_colors.astype(np.float32)),
            'labels': torch.from_numpy(sampled_labels.astype(np.int64))
        }

    def farthest_point_sampling(self, points):
        """最远点采样"""
        selected_indices = []
        distances = np.full(len(points), np.inf)

        # 随机选择第一个点
        current = np.random.randint(len(points))

        # 迭代选择最远点
        for _ in range(self.num_points):
            selected_indices.append(current)
            current_point = points[current]

            # 更新到已选择点集的最小距离
            dist = np.sum((points - current_point) ** 2, axis=1)
            distances = np.minimum(distances, dist)

            # 选择距离最大的点作为下一个点
            current = np.argmax(distances)

        return np.array(selected_indices)

    def __len__(self):
        return len(self.cached_chunks)

    def __getitem__(self, idx):
        return self.cached_chunks[idx]

    def normalize_points(self, points):
        """正规化点云坐标"""
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if max_dist > 0:
            points = points / max_dist
        return points

    def normalize_colors(self, colors):
        """确保颜色值在0-1范围内"""
        return np.clip(colors, 0, 1)


class BridgeValidationDataset(BridgePointCloudDataset):
    def __init__(self, data_dir, num_points=4096, block_size=1.0, overlap=0.3,
                 validation_ratio=0.3, seed=42):
        super().__init__(data_dir, num_points, block_size, overlap)

        # 设置随机种子
        np.random.seed(seed)

        # 随机采样validation_ratio的数据块
        total_chunks = len(self.cached_chunks)
        num_val_chunks = int(total_chunks * validation_ratio)

        # 随机选择索引
        selected_indices = np.random.choice(total_chunks, num_val_chunks, replace=False)
        self.cached_chunks = [self.cached_chunks[i] for i in selected_indices]

        logger.info(
            f"验证集创建完成，使用 {len(self.cached_chunks)} 个数据块 "
            f"(总共 {total_chunks} 个的 {validation_ratio * 100:.1f}%)"
        )
        logger.info(f"有效标签: {sorted(list(self.valid_labels))}")

    def __getitem__(self, idx):
        return self.cached_chunks[idx]
