import os

import laspy
import numba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class BriPCDMulti(Dataset):
    def __init__(self, data_dir, file_list=None, num_points=4096, transform=False, block_size=1.0,
                 sample_rate=0.5, logger=None):
        self.data_dir = data_dir
        self.num_points = num_points
        self.transform = transform
        self.block_size = block_size
        self.sample_rate = sample_rate
        self.logger = logger

        # 获取las文件列表
        if file_list is None:
            self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.las')]
        else:
            self.file_list = [f for f in file_list if f.endswith('.las')]

        if len(self.file_list) == 0:
            raise ValueError(f"No .las files found in {data_dir}")

        # 计算每个文件的采样次数
        self.file_sample_counts = []
        for filename in self.file_list:
            points_count = self._get_file_points_count(filename)
            sample_count = int(points_count * self.sample_rate / self.num_points)
            self.file_sample_counts.append(sample_count)

        # 计算总的数据块数量
        self.total_blocks = sum(self.file_sample_counts)

        # 创建文件索引映射
        self.block_to_file_map = []
        for file_idx, count in enumerate(self.file_sample_counts):
            self.block_to_file_map.extend([file_idx] * count)

        self.logger.info(f"Dataset initialized with {len(self.file_list)} files and {self.total_blocks} blocks")

    def _get_file_points_count(self, filename):
        """获取las文件中的点数量"""
        file_path = os.path.join(self.data_dir, filename)
        try:
            las = laspy.read(file_path)
            return len(las.points)
        except Exception as e:
            self.logger.error(f"Error reading point count from {filename}: {str(e)}")
            return 0

    def _get_files_hash(self, data_dir, file_list=None):
        """生成文件列表的哈希值，用于缓存标识"""
        import hashlib

        if file_list is None:
            files = sorted([f for f in os.listdir(data_dir) if f.endswith('.las')])
        else:
            files = sorted([f for f in file_list if f.endswith('.las')])

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

    def _load_las_file(self, filename):
        """加载单个las文件"""
        file_path = os.path.join(self.data_dir, filename)
        # print(f"Loading {file_path}")
        #self.logger.info(f"Loading {file_path}")

        try:
            las = laspy.read(file_path)

            # 获取点坐标
            points = np.vstack((las.x, las.y, las.z)).transpose()

            # 获取颜色信息（如果存在）
            if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                colors = np.vstack((las.red, las.green, las.blue)).transpose() / 65535.0  # 通常las文件的颜色范围是0-65535
            else:
                colors = np.ones_like(points)  # 如果没有颜色信息，使用默认值

            # 获取分类标签（如果存在）
            if hasattr(las, 'classification'):
                # 将SubFieldView转换为numpy数组
                labels = np.array(las.classification)
            else:
                labels = np.zeros(len(points), dtype=np.int64)

            # print(f"Loaded {len(points)} points from {filename}")
            #self.logger.info(f"Loaded {len(points)} points from {filename}")

            return points, colors, labels

        except Exception as e:
            # print(f"Error loading {filename}: {str(e)}")
            self.logger.error(f"Error loading {filename}: {str(e)}")

            return None, None, None

    def _preprocess_files(self):
        """预处理所有las文件"""
        all_blocks = []
        pbar = tqdm(self.file_list, desc="Preprocessing files", leave=True, position=0)

        for filename in pbar:

            self.logger.info(f"Processing {filename}")

            points, colors, labels = self._load_las_file(filename)

            if points is not None:
                # 分配点到块中
                blocks = self._assign_points_to_blocks(points, colors, labels, filename)
                self.logger.info(f"Created {len(blocks)} blocks")
                all_blocks.extend(blocks)

        self.logger.info(f"Total blocks created: {len(all_blocks)}")

        return all_blocks

    def _assign_points_to_blocks(self, points, colors, labels, filename):
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

        blocks = []
        num_pcd = len(points)
        pcd_iter = int(num_pcd * self.sample_rate / self.num_points)
        global_blocks = []
        local_blocks = []
        # 使用numpy的高效操作
        all_indices = np.arange(len(points))
        normal_points = self.normalize_points(points)

        self.logger.info(f"Total points: {num_pcd}, Iterations: {pcd_iter}")
        #print(f"Total points: {num_pcd}, Iterations: {pcd_iter}")

        for _ in range(pcd_iter):

            start_time = time.time()
            # 全局采样，随机采样到指定点数
            indices = np.random.choice(all_indices, self.num_points, replace=False)
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

            end_time=time.time()
            #print(f"Block creation time: {end_time - start_time:.2f} seconds")


        # 合并结果
        blocks = global_blocks + local_blocks

        return blocks


    def validate_dataset(self):
        """验证整个数据集，收集所有可能的标签"""
        self.logger.info("开始验证数据集...")
        for las_path in self.file_list:
            try:
                las = laspy.read(las_path)
                if hasattr(las, 'classification'):
                    unique_labels = np.unique(las.classification)
                    self.valid_labels.update(unique_labels)
            except Exception as e:
                self.logger.error(f"验证文件 {las_path} 时出错: {str(e)}")
                raise
        self.logger.info("数据集验证完成")

    def __len__(self):
        return self.total_blocks

    def __getitem__(self, idx):
        # 确定对应的文件
        file_idx = self.block_to_file_map[idx]
        filename = self.file_list[file_idx]

        # 加载点云数据
        points, colors, labels = self._load_las_file(filename)

        # 随机决定是全局采样还是局部采样
        if np.random.random() < 0.5:  # 50%概率使用全局采样
            # 全局采样
            indices = np.random.choice(len(points), self.num_points, replace=False)
            normal_points = self.normalize_points(points[indices])

            block = {
                'points': normal_points,
                'colors': colors[indices],
                'labels': labels[indices],
                'original_points': points[indices],
                'original_colors': colors[indices],
                'file_name': filename,
                'indices': indices
            }
        else:
            # 局部采样
            max_attempts = 10
            for _ in range(max_attempts):
                center = points[np.random.choice(len(points))][:3]
                block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
                block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]

                # 使用numba加速的函数查找块内的点
                block_indices = self._find_points_in_block(points, block_min, block_max)

                if len(block_indices) >= self.num_points:
                    indices = np.random.choice(block_indices, self.num_points, replace=False)
                    normal_points = self.normalize_points(points[indices])

                    block = {
                        'points': normal_points,
                        'colors': colors[indices],
                        'labels': labels[indices],
                        'original_points': points[indices],
                        'original_colors': colors[indices],
                        'file_name': filename,
                        'indices': indices
                    }
                    break
            else:
                # 如果多次尝试都失败，退回到全局采样
                indices = np.random.choice(len(points), self.num_points, replace=False)
                normal_points = self.normalize_points(points[indices])

                block = {
                    'points': normal_points,
                    'colors': colors[indices],
                    'labels': labels[indices],
                    'original_points': points[indices],
                    'original_colors': colors[indices],
                    'file_name': filename,
                    'indices': indices
                }

        # 应用数据增强
        if self.transform:
            block['points'], block['colors'] = self.apply_transform(
                block['points'], block['colors'])

        return {
            'points': block['points'].astype(np.float32),
            'colors': block['colors'].astype(np.float32),
            'labels': block['labels'].astype(np.int64),
            'original_points': block['original_points'].astype(np.float32),
            'original_colors': block['original_colors'].astype(np.float32),
            'file_name': block['file_name'],
            'indices': block['indices']
        }

    @staticmethod
    @numba.jit(nopython=True)
    def _find_points_in_block(points, block_min, block_max, z_threshold=2.0):
        """使用numba加速的块内点查找函数"""
        mask = np.zeros(len(points), dtype=np.bool_)
        for i in range(len(points)):
            if (block_min[0] <= points[i, 0] <= block_max[0] and
                    block_min[1] <= points[i, 1] <= block_max[1]):
                z_center = (block_min[2] + block_max[2]) / 2
                if abs(points[i, 2] - z_center) <= z_threshold:
                    mask[i] = True
        return np.where(mask)[0]

    def apply_transform(self, points, colors, keep_original=True):
        """改进的数据增强函数"""
        if not self.transform:
            return points, colors

        # 如果需要保留原始数据，创建副本
        if keep_original:
            points = points.copy()
            if colors is not None:
                colors = colors.copy()

        # 随机旋转 - 这个是合理的，不会改变数据范围
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        points = np.dot(points, rotation_matrix)

        # 随机平移 - 可以调整范围或在变换后重新归一化
        translation = np.random.uniform(0.01, 0.1, size=(1, 3))  # 缩小范围
        points += translation

        # 随机缩放 - 可以调整范围或在变换后重新归一化
        scale = np.random.uniform(0.9, 1.1)  # 缩小范围
        points *= scale

        # 可选：重新归一化点云数据
        #points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))

        # 随机抖动颜色 - 已经有clip操作，这个是合理的
        if colors is not None:
            color_noise = np.random.normal(0, 0.02, colors.shape)
            colors = np.clip(colors + color_noise, 0, 1)

        return points, colors


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    import logging


    # 测试数据集加载和可视化
    def visualize_block(dataset, block_idx=0):
        print(f"\n正在可视化第 {block_idx} 个数据块...")

        # 获取一个数据块
        start_time = time.time()
        data = dataset[block_idx]
        load_time = time.time() - start_time
        print(f"数据加载时间: {load_time:.2f} 秒")

        # 打印数据块信息
        points = data['points']
        colors = data['colors']
        labels = data['labels']

        print("\n数据块统计信息:")
        print(f"点数: {len(points)}")
        print(f"点云范围:")
        print(f"X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
        print(labels)

        label_counts = np.bincount(labels, minlength=5)

        # 打印每个标签的数量
        for i in range(5):
            print(f"Label {i}: {label_counts[i]}")

        # 创建3D可视化图
        fig = plt.figure(figsize=(15, 5))

        # 1. 使用坐标显示
        ax1 = fig.add_subplot(131, projection='3d')
        scatter1 = ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                               c=labels,
                               cmap='tab20', s=1)
        ax1.set_title('PCD coordinate view')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # 2. 使用颜色显示
        ax2 = fig.add_subplot(132, projection='3d')
        scatter2 = ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
                               c=colors,  # RGB颜色
                               s=1)
        ax2.set_title('PCD color view')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        # 3. 使用标签显示
        ax3 = fig.add_subplot(133, projection='3d')
        scatter3 = ax3.scatter(points[:, 0], points[:, 1], points[:, 2],
                               c=labels,
                               cmap='tab20', s=1)
        ax3.set_title('PCD label view')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')

        # 添加颜色条
        plt.colorbar(scatter1, ax=ax1, label='Labels')
        plt.colorbar(scatter3, ax=ax3, label='Labels')

        plt.tight_layout()
        plt.show()

        # 打印数据形状
        print("\n数据形状:")
        print(f"Points shape: {points.shape}")
        print(f"Colors shape: {colors.shape}")
        print(f"Labels shape: {labels.shape}")

        return data

        # 创建数据加载器


    def get_logger():
        logger = logging.getLogger('BriPCDMulti_dataset_5class')
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger


    # train_dataset = BriPCDMulti(
    #     data_dir='../../data/CB/section/train/',
    #     num_points=4096,
    #     block_size=1.0,
    #     sample_rate=0.4,
    #     logger=get_logger(),
    #     transform=True
    # )

    val_dataset = BriPCDMulti(
        data_dir='../../data/CB/section/val/',
        num_points=4096,
        block_size=1.0,
        sample_rate=0.4,
        logger=get_logger()
    )

    # # DataLoader的使用方式完全不变
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=16,
    #     shuffle=True,
    #     num_workers=0,
    #     pin_memory=True
    # )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4
    )

    #data = train_dataset[200]
    # labels = data['labels']
    # # 统计每个标签的数量
    # label_counts = np.bincount(labels, minlength=5)
    #
    # # 打印每个标签的数量
    # for i in range(5):
    #     print(f"Label {i}: {label_counts[i]}")

    # 可视化第一个数据块
    #block_data = visualize_block(train_dataset, block_idx=1)