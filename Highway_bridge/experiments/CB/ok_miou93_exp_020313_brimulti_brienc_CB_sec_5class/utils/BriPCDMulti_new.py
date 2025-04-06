import hashlib
import logging
import multiprocessing
import os

import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader


class BriPCDMulti(Dataset):
    def __init__(self, data_dir, file_list=None, num_points=4096, transform=False, block_size=1.0,
                 sample_rate=0.5, logger=None):
        """
        Args:
            data_dir (str): 数据目录
            num_points (int): 每个点云块的大小
            transform (bool): 是否进行数据增强
        """
        self.data_dir = data_dir
        self.num_points = num_points
        self.transform = transform
        self.block_size = block_size
        self.sample_rate = sample_rate
        self.logger = logger if logger else logging.getLogger(__name__)

        # 定义缓存文件夹
        self.cache_dir = os.path.join(self.data_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        # 生成缓存标识
        cache_id = self._get_cache_id(data_dir, file_list)

        # 记录所有块的信息
        self.block_info = []

        # 获取文件列表
        if file_list is None:
            self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.h5')]
        else:
            self.file_list = [f for f in file_list if f.endswith('.h5')]

        if len(self.file_list) == 0:
            raise ValueError(f"No .h5 files found in {data_dir}")

        self.logger.info(f"Found {len(self.file_list)} h5 files:")
        for f in self.file_list:
            self.logger.info(f"  - {f}")

        # 预处理所有文件
        self._preprocess_files(cache_id)

    def _get_cache_id(self, data_dir, file_list=None):
        """生成缓存标识，用于缓存文件夹命名"""
        if file_list is None:
            files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])
        else:
            files = sorted([f for f in file_list if f.endswith('.h5')])

        # 将文件名和最后修改时间组合起来生成哈希
        content = []
        for f in files:
            file_path = os.path.join(data_dir, f)
            mtime = os.path.getmtime(file_path)
            content.append(f"{f}_{mtime}")

        content = "_".join(content)
        return hashlib.md5(content.encode()).hexdigest()[:8]

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

    def _load_hdf5_file(self, hdf5_filename):
        """加载 HDF5 文件"""
        file_path = os.path.join(self.data_dir, hdf5_filename)
        self.logger.info(f"Loading {file_path}")
        with h5py.File(file_path, 'r') as f:
            points = np.array(f['points'])
            colors = np.array(f['colors'])
            labels = np.array(f['labels'])

        self.logger.info(f"Loaded {len(points)} points from {hdf5_filename}")
        return points, colors, labels

    def _preprocess_file(self, args):
        """预处理单个文件"""
        filename, cache_id = args
        cache_file = os.path.join(self.cache_dir, f"{filename}_{cache_id}.npz")
        block_infos = []

        if os.path.exists(cache_file):
            self.logger.info(f"Cache exists for {filename}, skipping preprocessing.")
            # 如果缓存存在，加载块数量信息
            data = np.load(cache_file)
            num_blocks = data['num_blocks']
            for i in range(num_blocks):
                block_file = os.path.join(self.cache_dir, f"{filename}_{cache_id}_block_{i}.npz")
                block_infos.append((block_file, i))
            return block_infos

        self.logger.info(f"Processing {filename}")

        points, colors, labels = self._load_hdf5_file(filename)

        if points is not None:
            # 分配点到块中
            blocks = self._assign_points_to_blocks(points, colors, labels, filename)

            # 将每个块的数据保存到缓存文件中
            num_blocks = len(blocks)
            for i, block in enumerate(blocks):
                block_file = os.path.join(self.cache_dir, f"{filename}_{cache_id}_block_{i}.npz")
                np.savez_compressed(
                    block_file,
                    points=block['points'].astype(np.float32),
                    colors=block['colors'].astype(np.float32),
                    labels=block['labels'].astype(np.int64),
                    original_points=block['original_points'].astype(np.float32),
                    original_colors=block['original_colors'].astype(np.float32),
                    indices=block['indices'].astype(np.int64)
                )
                block_infos.append((block_file, i))

            # 保存块数量信息
            np.savez_compressed(cache_file, num_blocks=num_blocks)

            self.logger.info(f"Created {num_blocks} blocks for {filename}")
            return block_infos
        else:
            self.logger.warning(f"No points loaded for {filename}")
            return []

    def _preprocess_files(self, cache_id):
        """并行预处理所有文件"""
        all_args = [(filename, cache_id) for filename in self.file_list]

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(self._preprocess_file, all_args)

        for block_infos in results:
            self.block_info.extend(block_infos)

        self.logger.info(f"Total blocks created: {len(self.block_info)}")

    def _assign_points_to_blocks(self, points, colors, labels, filename):
        """分配点到块中"""
        # 复制您的原始 _assign_points_to_blocks 方法的实现

        # 以下是您的原始方法，将其完整粘贴到这里
        import numba
        import numpy as np
        import time

        @numba.jit(nopython=True)
        def find_points_in_block(points, block_min, block_max, z_threshold=2.0):
            mask = np.zeros(len(points), dtype=np.bool_)
            for i in range(len(points)):
                if (block_min[0] <= points[i, 0] <= block_max[0] and
                        block_min[1] <= points[i, 1] <= block_max[1]):
                    z_center = (block_min[2] + block_max[2]) / 2
                    if abs(points[i, 2] - z_center) <= z_threshold:
                        mask[i] = True
            return np.where(mask)[0]

        def check_block_validity(block_indices, min_labels=1):
            """
            检查采样块是否满足要求
            """
            if len(block_indices) < self.num_points:
                return False

            block_labels = labels[block_indices]
            unique_labels = np.unique(block_labels)
            return len(unique_labels) >= min_labels

        def weighted_stratified_sampling(points, labels, colors, num_points, desired_class_proportions):
            """
            根据期望的类别比例进行加权分层采样
            Args:
                points: 点云坐标数组 (N, 3)
                labels: 标签数组 (N,)
                colors: 颜色数组 (N, 3)
                num_points: 每个块的总点数
                desired_class_proportions: 期望的类别比例字典
            Returns:
                selected_indices: 选择的点的索引数组
            """
            unique_classes = np.unique(labels)
            selected_indices = []
            total_proportion = sum(desired_class_proportions.values())

            # 规范化类别比例，使其总和为1
            normalized_proportions = {k: v / total_proportion for k, v in desired_class_proportions.items()}

            # 计算每个类别需要采样的点数
            desired_num_points_per_class = {}
            for cls in unique_classes:
                proportion = normalized_proportions.get(cls, 0)
                desired_num_points_per_class[cls] = int(proportion * num_points)

            # 调整采样点数，使总和为 num_points
            total_desired = sum(desired_num_points_per_class.values())
            diff = num_points - total_desired
            if diff != 0:
                # 将差值加到最大的类别上
                max_class = max(desired_num_points_per_class, key=desired_num_points_per_class.get)
                desired_num_points_per_class[max_class] += diff

            # 对每个类别进行采样
            for cls in unique_classes:
                class_indices = np.where(labels == cls)[0]
                n_samples = desired_num_points_per_class.get(cls, 0)
                if n_samples <= 0 or len(class_indices) == 0:
                    continue
                if len(class_indices) >= n_samples:
                    selected = np.random.choice(class_indices, n_samples, replace=False)
                else:
                    # 如果样本不足，允许重复采样
                    selected = np.random.choice(class_indices, n_samples, replace=True)
                selected_indices.extend(selected)

            # 转换为 numpy 数组并打乱顺序
            selected_indices = np.array(selected_indices)
            np.random.shuffle(selected_indices)
            return selected_indices

        blocks = []
        global_blocks = []
        local_blocks = []
        num_pcd = len(points)
        pcd_iter = int(num_pcd * self.sample_rate / self.num_points)
        normal_points = self.normalize_points(points)

        self.logger.info(f"Total points: {num_pcd}, Iterations: {pcd_iter}")

        # 定义期望的类别比例
        desired_class_proportions = {
            0: 0.05,  # 减少 class 0 的采样
            1: 0.2,
            2: 0.2,
            3: 0.2,
            4: 0.35  # 增加 class 4 的采样
        }

        for _ in range(pcd_iter):
            # 使用加权分层采样
            indices = weighted_stratified_sampling(
                points=points,
                labels=labels,
                colors=colors,
                num_points=self.num_points,
                desired_class_proportions=desired_class_proportions
            )

            block = {
                'points': normal_points[indices],
                'colors': colors[indices],
                'labels': labels[indices],
                'original_points': points[indices],  # 保存原始坐标
                'original_colors': colors[indices],
                'file_name': filename,  # 添加文件名
                'indices': indices  # 保存原始索引
            }
            global_blocks.append(block)

            # 局部采样
            # 随机选择一个中心点
            n_points = points.shape[0]  # N
            center = points[np.random.choice(n_points)][: 3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]

            # 查找块内的点
            block_indices = find_points_in_block(points, block_min, block_max)
            # 检查是否满足采样条件
            if check_block_validity(block_indices):
                sampled_indices = np.random.choice(block_indices, self.num_points, replace=False)
                block = {
                    'points': normal_points[sampled_indices],
                    'colors': colors[sampled_indices],
                    'labels': labels[sampled_indices],
                    'original_points': points[sampled_indices],  # 保存原始坐标
                    'original_colors': colors[sampled_indices],
                    'file_name': filename,  # 添加文件名
                    'indices': sampled_indices  # 保存原始索引
                }
                local_blocks.append(block)

            end_time = time.time()

        # 合并结果
        blocks = global_blocks + local_blocks

        return blocks

    def __len__(self):
        return len(self.block_info)

    def __getitem__(self, idx):
        block_file, block_idx = self.block_info[idx]

        # 加载块数据
        data = np.load(block_file)
        points = data['points']
        colors = data['colors']
        labels = data['labels']
        original_points = data['original_points']
        original_colors = data['original_colors']
        indices = data['indices']

        # 应用数据增强
        if self.transform:
            points, colors = self.apply_transform(points, colors)

        return {
            'points': points.astype(np.float32),
            'colors': colors.astype(np.float32),
            'labels': labels.astype(np.int64),
            'original_points': original_points.astype(np.float32),
            'original_colors': original_colors.astype(np.float32),
            'file_name': os.path.basename(block_file),
            'indices': indices.astype(np.int64)
        }

    def apply_transform(self, points, colors, keep_original=True):
        """数据增强函数"""
        if not self.transform:
            return points, colors

        # 如果需要保留原始数据，创建副本
        if keep_original:
            points = points.copy()
            if colors is not None:
                colors = colors.copy()

        # 随机旋转
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        points = np.dot(points, rotation_matrix)

        # 随机平移
        translation = np.random.uniform(0.01, 0.1, size=(1, 3))
        points += translation

        # 随机缩放
        scale = np.random.uniform(0.9, 1.1)
        points *= scale

        # 随机抖动颜色
        if colors is not None:
            color_noise = np.random.normal(0, 0.02, colors.shape)
            colors = np.clip(colors + color_noise, 0, 1)

        return points, colors

# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 初始化数据集
    dataset = BriPCDMulti(
        data_dir='../../data/CB/all/train',
        num_points=4096,
        block_size=1,
        sample_rate=0.2, #0.2 是只有全局
        transform=True,
        logger=logger
    )

    dataset = BriPCDMulti(
        data_dir='../../data/CB/all/val',
        num_points=4096,
        block_size=1,
        sample_rate=0.2, #0.2 是只有全局
        logger=logger
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
        print(f"Batch colors shape: {batch['colors'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")
        break
