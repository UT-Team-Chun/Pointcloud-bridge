import os
import laspy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import hashlib
import numba
import time
import h5py

class BridgeDataset(Dataset):
    def __init__(self, data_dir, file_list=None, num_points=4096, block_size=5.0,
                 sample_stride=0.5, transform=False, cache=True, logger=None):
        """
        优化后的桥梁点云数据集类

        Args:
            data_dir (str): 数据目录路径
            num_points (int): 每个样本的点数 (默认: 4096)
            block_size (float): 空间分块尺寸 (米)
            sample_stride (float): 采样步长 (块尺寸的比率)
            transform (bool): 是否启用数据增强
            cache (bool): 是否使用缓存
            logger: 日志记录器
        """
        self.data_dir = data_dir
        self.num_points = num_points
        self.block_size = block_size
        self.sample_stride = block_size * sample_stride
        self.transform = transform
        self.cache = cache
        self.logger = logger or logging.getLogger(__name__)

        # 初始化点云缓存
        self.point_clouds = []
        self.point_clouds_raw = []
        self.file_metadata = []

        # 缓存配置
        self.cache_dir = os.path.join(data_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        # 加载或处理数据
        self._load_or_process_data(file_list)

    def _load_or_process_data(self, file_list):
        """数据加载和处理主逻辑"""
        file_list = self._validate_file_list(file_list)
        cache_hash = self._generate_cache_hash(file_list)
        cache_path = os.path.join(self.cache_dir, f'dataset_{cache_hash}.npz')

        if self.cache and os.path.exists(cache_path):
            self.logger.info(f"Loading cached data from {cache_path}")
            self._load_cached_data(cache_path)
        else:
            self.logger.info("Processing raw data...")
            self._process_raw_data(file_list)
            if self.cache:
                self._save_data_to_cache(cache_path)

        self._precompute_blocks()

    def _validate_file_list(self, file_list):
        """验证并获取有效文件列表"""
        if file_list is None:
            file_list = [f for f in os.listdir(self.data_dir) if f.endswith('.h5')]
        else:
            file_list = [f for f in file_list if f.endswith('.h5')]

        if not file_list:
            raise ValueError(f"No LAS files found in {self.data_dir}")
        return file_list

    def _generate_cache_hash(self, file_list):
        """生成缓存哈希值"""
        hash_str = ''.join(sorted(file_list)) + \
                   f"{self.num_points}_{self.block_size}_{self.sample_stride}"
        return hashlib.md5(hash_str.encode()).hexdigest()[:12]

    def _process_raw_data(self, file_list):
        """处理原始LAS文件"""
        for filename in tqdm(file_list, desc="Processing LAS files"):
            file_path = os.path.join(self.data_dir, filename)
            try:
                with h5py.File(file_path, 'r') as f:
                    points = np.array(f['points'])
                    colors = np.array(f['colors'])
                    labels = np.array(f['labels'])

                # 存储元数据
                self.point_clouds_raw.append({
                    'points': points.astype(np.float32),
                    'colors': colors.astype(np.float32),
                    'labels': labels.astype(np.int64),
                    'bounds': self._compute_bounds(points)
                })
                # 空间归一化
                points = self._normalize_points(points)

                self.point_clouds.append({
                    'points': points.astype(np.float32),
                    'colors': colors.astype(np.float32),
                    'labels': labels.astype(np.int64),
                    'bounds': self._compute_bounds(points)
                })
                self.file_metadata.append({
                    'filename': filename,
                    'num_points': len(points),
                    'timestamp': time.time()
                })

            except Exception as e:
                self.logger.error(f"Error processing {filename}: {str(e)}")
                continue

    def _get_colors(self, las):
        """提取颜色信息"""
        if all(hasattr(las, c) for c in ['red', 'green', 'blue']):
            return np.vstack([las.red, las.green, las.blue]).T / 65535.0
        return np.ones((len(las.points), 3), dtype=np.float32)

    def _get_labels(self, las):
        """提取分类标签"""
        return las.classification if hasattr(las, 'classification') else np.zeros(len(las.points), dtype=np.int64)

    def _normalize_points(self, points):
        """点云归一化"""
        # 保留原始坐标
        self.original_points = points.copy()

        centroid = np.mean(points, axis=0)
        points -= centroid
        max_dist = np.max(np.linalg.norm(points, axis=1))
        return points / max_dist if max_dist > 0 else points

    def _compute_bounds(self, points):
        """计算点云包围盒"""
        return np.array([points.min(axis=0), points.max(axis=0)])

    def _precompute_blocks(self):
        """预计算有效区块"""
        self.blocks = []
        for pc_idx, pc in enumerate(self.point_clouds_raw):
            bounds = pc['bounds']
            grid_steps = np.ceil((bounds[1] - bounds[0]) / self.sample_stride).astype(int)

            for x in range(grid_steps[0]):
                for y in range(grid_steps[1]):
                    min_corner = bounds[0] + [x * self.sample_stride, y * self.sample_stride, 0]
                    max_corner = min_corner + [self.block_size, self.block_size, bounds[1][2]]

                    # 使用numba加速的块查询
                    block_indices = self._find_points_in_block(pc['points'], min_corner, max_corner)
                    if len(block_indices) >= self.num_points // 2:
                        self.blocks.append({
                            'pc_index': pc_idx,
                            'indices': block_indices,
                            'center': (min_corner + max_corner) / 2
                        })

    @staticmethod
    @numba.jit(nopython=True)
    def _find_points_in_block(points, min_corner, max_corner):
        """快速块查询函数"""
        mask = np.zeros(len(points), dtype=np.bool_)
        for i in range(len(points)):
            if (min_corner[0] <= points[i, 0] <= max_corner[0] and
                    min_corner[1] <= points[i, 1] <= max_corner[1] and
                    min_corner[2] <= points[i, 2] <= max_corner[2]):
                mask[i] = True
        return np.where(mask)[0]

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        block = self.blocks[idx]
        pc = self.point_clouds[block['pc_index']]

        # 分层随机采样
        indices = self._stratified_sampling(
            pc['labels'][block['indices']],
            block['indices'],
            self.num_points
        )

        points = pc['points'][indices].astype(np.float32)  # 确保是 float32
        colors = pc['colors'][indices].astype(np.float32)  # 确保是 float32
        labels = pc['labels'][indices]

        # 数据增强
        if self.transform:
            points, colors = self._apply_augmentation(points, colors)

        # 转换为张量
        return {
            'points': torch.from_numpy(points).float(),  # 转换为 float32
            'features': torch.from_numpy(colors).float(),  # 转换为 float32
            'labels': torch.from_numpy(labels)
        }

    def _stratified_sampling(self, labels, indices, target_num):
        """改进的分层采样"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        samples = []

        # 按类别比例采样
        for label, count in zip(unique_labels, counts):
            label_indices = indices[labels == label]
            sample_num = max(1, int(target_num * count / len(indices)))
            samples.append(np.random.choice(label_indices, sample_num, replace=True))

        # 合并并补充不足点数
        sampled = np.concatenate(samples)
        if len(sampled) < target_num:
            supplemented = np.random.choice(indices, target_num - len(sampled), replace=True)
            sampled = np.concatenate([sampled, supplemented])

        return np.random.permutation(sampled)[:target_num]

    def _apply_augmentation(self, points, colors):
        """优化的数据增强"""
        # 随机旋转
        angle = np.random.uniform(0, 2 * np.pi)
        rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        points = points @ rot_matrix

        # 随机缩放
        scale = np.random.uniform(0.9, 1.1)
        points *= scale

        # 颜色抖动
        color_noise = np.random.normal(0, 0.05, colors.shape)
        colors = np.clip(colors + color_noise, 0, 1)

        return points, colors

    def _save_data_to_cache(self, cache_path):
        """保存处理后的数据到缓存"""

        def pad_array(array, target_size, pad_value=0):
            """将数组填充到目标大小"""
            if len(array) >= target_size:
                return array[:target_size]
            else:
                padding = np.full((target_size - len(array), array.shape[1]), pad_value, dtype=array.dtype)
                return np.vstack((array, padding))

        points = [pad_array(pc['points'], self.num_points) for pc in self.point_clouds]
        colors = [pad_array(pc['colors'], self.num_points) for pc in self.point_clouds]
        labels = [pad_array(pc['labels'], self.num_points, pad_value=-1) for pc in self.point_clouds]  # 用 -1 表示无效标签

        np.savez_compressed(
            cache_path,
            points=np.array(points),
            colors=np.array(colors),
            labels=np.array(labels),
            metadata=self.file_metadata
        )

    def _load_cached_data(self, cache_path):
        """从缓存加载数据"""
        data = np.load(cache_path, allow_pickle=True)
        for points, colors, labels in zip(data['points'], data['colors'], data['labels']):
            self.point_clouds.append({
                'points': points,
                'colors': colors,
                'labels': labels,
                'bounds': self._compute_bounds(points)
            })
        self.file_metadata = data['metadata'].tolist()


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 初始化数据集
    dataset = BridgeDataset(
        data_dir='../../data/CB/all/val',
        num_points=4096,
        block_size=1,
        sample_stride=0.5,
        transform=True,
        cache=True
    )

    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 测试批次加载
    for batch in dataloader:
        print(f"Batch points shape: {batch['points'].shape}")
        print(f"Batch features shape: {batch['features'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")
        break