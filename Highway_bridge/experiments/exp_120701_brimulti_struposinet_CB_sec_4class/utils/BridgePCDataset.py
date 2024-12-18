import os

import laspy
import numpy as np
from torch.utils.data import Dataset


class BridgePointCloudDataset(Dataset):
    def __init__(self,
                 data_dir,
                 file_list=None,  # 可以指定具体的文件列表
                 num_points=4096,
                 h_block_size=1.0,
                 v_block_size=1.0,
                 h_stride=0.5,
                 v_stride=0.5,
                 min_points=100,
                 transform=None,
                 logger=None
                 ):
        """
        初始化桥梁点云数据集
        Args:
            data_dir: 数据目录，包含las文件
            file_list: 指定的las文件列表，如果为None则读取目录下所有las文件
            h_block_size: 水平方向块大小
            v_block_size: 垂直方向块大小
            h_stride: 水平方向滑动步长
            v_stride: 垂直方向滑动步长
            min_points: 每个块最少需要包含的点数
            transform: 数据增强转换
        """
        self.data_dir = data_dir
        self.num_points = num_points
        self.h_block_size = h_block_size
        self.v_block_size = v_block_size
        self.h_stride = h_stride
        self.v_stride = v_stride
        self.min_points = min_points
        self.transform = transform
        self.logger = logger
        
        # 获取las文件列表
        if file_list is None:
            self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.las')]
        else:
            self.file_list = [f for f in file_list if f.endswith('.las')]

        if len(self.file_list) == 0:
            raise ValueError(f"No .las files found in {data_dir}")

        self.logger.info(f"Found {len(self.file_list)} las files:")

        for f in self.file_list:
            print(f"  - {f}")

        # 预处理所有文件
        self.blocks = self._preprocess_files()

    def _load_las_file(self, filename):
        """加载单个las文件"""
        file_path = os.path.join(self.data_dir, filename)
        #print(f"Loading {file_path}")
        self.logger.info(f"Loading {file_path}")

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

            #print(f"Loaded {len(points)} points from {filename}")
            self.logger.info(f"Loaded {len(points)} points from {filename}")

            return points, colors, labels

        except Exception as e:
            #print(f"Error loading {filename}: {str(e)}")
            self.logger.error(f"Error loading {filename}: {str(e)}")

            return None, None, None

    def _assign_points_to_blocks(self, points, colors, labels):

        """将点分配到不同的块中"""
        #print(f"Processing data - Points: {points.shape}, Colors: {colors.shape}, Labels: {labels.shape}")
        self.logger.info(f"Processing data - Points: {points.shape}, Colors: {colors.shape}, Labels: {labels.shape}")

        # 计算点云的边界
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        print(f"全局点云范围:")
        print(f"最小坐标: {min_coords}")
        print(f"最大坐标: {max_coords}")
        print(f"点云总范围: {max_coords - min_coords}")

        # 打印block和stride的设置
        print(f"\n块大小设置:")
        print(f"h_block_size: {self.h_block_size}")
        print(f"v_block_size: {self.v_block_size}")
        print(f"h_stride: {self.h_stride}")
        print(f"v_stride: {self.v_stride}")

        ranges = max_coords - min_coords
        print(f"Ranges: {ranges}")
        ranges[ranges < 1e-6] = 1.0
        normalized_points = (points - min_coords) / ranges

        # 计算每个点所属的网格索引
        grid_indices = np.floor((points - min_coords) / [self.h_stride, self.h_stride, self.v_stride]).astype(int)

        # 打印网格索引的范围
        print(f"\n网格索引范围:")
        print(f"最小网格索引: {np.min(grid_indices, axis=0)}")
        print(f"最大网格索引: {np.max(grid_indices, axis=0)}")

        # 计算网格在每个维度上的范围
        grid_min = np.min(grid_indices, axis=0)  # 最小网格索引
        grid_max = np.max(grid_indices, axis=0)  # 最大网格索引
        grid_size = grid_max - grid_min + 1  # 每个维度的网格数量
        print(f"\n网格维度信息:")
        print(f"X轴网格数量: {grid_size[0]} (索引范围: {grid_min[0]} 到 {grid_max[0]})")
        print(f"Y轴网格数量: {grid_size[1]} (索引范围: {grid_min[1]} 到 {grid_max[1]})")
        print(f"Z轴网格数量: {grid_size[2]} (索引范围: {grid_min[2]} 到 {grid_max[2]})")
        print(f"理论总网格数: {np.prod(grid_size)}")


        # 使用字典收集每个网格中的点
        grid_dict = {}
        for i, grid_idx in enumerate(grid_indices):
            key = tuple(grid_idx)
            if key not in grid_dict:
                grid_dict[key] = []
            grid_dict[key].append(i)

        print(f"实际非空网格数: {len(grid_dict)}")
        print(f"网格占用率: {len(grid_dict) / np.prod(grid_size):.2%}")
        print(f"\n总共的网格数量: {len(grid_dict)}")

        blocks = []
        #norm_points = self.normalize_points(points)

        # 检查第一个block的详细信息
        first_block = True

        # 处理每个非空的网格
        for grid_idx, indices in grid_dict.items():
            indices = np.array(indices)
            if len(indices) >= self.min_points:
                # 计算块的中心点
                center = min_coords + np.array(grid_idx) * [self.h_stride, self.h_stride, self.v_stride] + \
                         [self.h_block_size / 2, self.h_block_size / 2, self.v_block_size / 2]

                block_points = normalized_points[indices]
                #block_colors = colors[indices]
                #block_labels = labels[indices]

                # 将点坐标归一化到块的中心
                #normalized_block = block_points - center

                valid_mask = np.all(
                    (points[indices] >= center - [self.h_block_size / 2, self.h_block_size / 2, self.v_block_size / 2]) &
                    (points[indices] < center + [self.h_block_size / 2, self.h_block_size / 2, self.v_block_size / 2]),
                    axis=1
                )

                if np.sum(valid_mask) >= self.min_points:
                    valid_indices = indices[valid_mask]
                    if first_block:
                        print(f"\n第一个block的详细信息:")
                        print(f"网格索引: {grid_idx}")
                        print(f"中心点: {center}")
                        print(f"block内点的范围:")
                        block_min = np.min(points[valid_indices], axis=0)
                        block_max = np.max(points[valid_indices], axis=0)
                        print(f"最小坐标: {block_min}")
                        print(f"最大坐标: {block_max}")
                        print(f"block实际大小: {block_max - block_min}")
                        print(f"有效点数量: {np.sum(valid_mask)}")
                        first_block = False

                    blocks.append({
                    'points': block_points[valid_mask],
                    'colors': colors[valid_indices],
                    'labels': labels[valid_indices],
                    'center': center,
                    'indices': indices
                    #'normalized_points': normalized_block
                })

        self.logger.info(f"Created {len(blocks)} blocks")

        return blocks

    def normalize_points(self, points):
        """正规化点云坐标"""
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)

        # 处理每个维度的范围
        ranges = max_coords - min_coords
        # 避免除零，同时保持维度特征
        ranges[ranges < 1e-6] = 1.0

        normalized_points = (points - min_coords) / ranges

        return normalized_points

    def _preprocess_files(self):
        """预处理所有las文件"""
        all_blocks = []

        for filename in self.file_list:
            points, colors, labels = self._load_las_file(filename)

            if points is not None:
                # 分配点到块中
                blocks = self._assign_points_to_blocks(points, colors, labels)
                all_blocks.extend(blocks)

        print(f"Total blocks created: {len(all_blocks)}")
        self.logger.info(f"Total blocks created: {len(all_blocks)}")

        return all_blocks

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

    def __len__(self):
        """返回数据集中块的数量"""
        return len(self.blocks)

    def __getitem__(self, idx):
        """获取指定索引的数据块"""
        block = self.blocks[idx]

        # 转换为所需的数据格式
        points = block['points'].astype(np.float32)
        colors = block['colors'].astype(np.float32)
        labels = block['labels'].astype(np.int64)

        # 统一点数
        if self.num_points is not None:
            choice = self.density_aware_sample(points, self.num_points)

        points = points[choice]
        colors = colors[choice]
        labels = labels[choice]

        # 应用数据增强
        if self.transform:
            points, colors = self.apply_transform(points, colors)

        return {
            'points': points.astype(np.float32),
            'colors': colors.astype(np.float32),
            'labels': labels.astype(np.int64),
            'center': block['center']
        }

    def density_aware_sample(self, points, npoint):
        """
        密度感知采样
        Args:
            points: (N, 3) 输入点云
            npoint: 目标采样点数
        Returns:
            choice: (npoint,) 采样点的索引
        """
        N, D = points.shape

        if N <= npoint:
            # 如果原始点数小于目标点数，直接重复采样
            choice = np.random.choice(N, size=npoint, replace=True)
            return choice

        # 1. 计算每个点的局部密度
        from sklearn.neighbors import NearestNeighbors
        k = min(32, N)  # 近邻数量
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(points)
        distances, _ = nbrs.kneighbors(points)

        # 计算局部密度（使用平均距离的倒数）
        densities = 1 / (np.mean(distances, axis=1) + 1e-8)

        # 2. 根据密度计算采样概率
        probabilities = densities / np.sum(densities)

        # 确保概率和为1
        probabilities = probabilities / np.sum(probabilities)

        # 3. 采样策略：结合FPS和密度感知采样
        # 首先使用FPS采样一部分点（占总数的70%）
        fps_npoint = int(0.7 * npoint)
        fps_indices = self.farthest_point_sample(points, fps_npoint)

        # 剩余点使用密度感知随机采样
        remaining_npoint = npoint - fps_npoint

        # 更新概率（已选择的点概率设为0）
        mask = np.ones(N, dtype=bool)
        mask[fps_indices] = False
        probabilities[~mask] = 0

        # 重新归一化概率
        if np.sum(probabilities) > 0:
            probabilities = probabilities / np.sum(probabilities)
        else:
            # 如果所有概率都为0，使用均匀分布
            probabilities[mask] = 1.0 / np.sum(mask)

        # 采样剩余点
        density_indices = np.random.choice(
            N,
            size=remaining_npoint,
            replace=False,
            p=probabilities
        )

        # 合并两种采样的结果
        choice = np.concatenate([fps_indices, density_indices])

        return choice

    def farthest_point_sample(self, points, npoint):
        """
        使用FPS（最远点采样）选择点
        """
        N, D = points.shape
        centroids = np.zeros(npoint, dtype=np.int64)
        distance = np.ones(N) * 1e10
        farthest = np.random.randint(0, N)

        for i in range(npoint):
            centroids[i] = farthest
            centroid = points[farthest]
            dist = np.sum((points - centroid) ** 2, axis=1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance)

        return centroids


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    from logger_config import get_logger

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
        center = data['center']

        print("\n数据块统计信息:")
        print(f"点数: {len(points)}")
        print(f"中心点坐标: {center}")
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


    specific_file = ['bridge-7.las']
    # 创建数据集实例
    dataset = BridgePointCloudDataset(
        data_dir='../data/fukushima/onepart/val',  # 替换为你的数据路径
        file_list=specific_file,
        num_points=4096,
        h_block_size=5,
        v_block_size=2,
        h_stride=4,
        v_stride=3,
        min_points=100,
        logger = get_logger()
    )

    # data = dataset[200]
    # labels = data['labels']
    # # 统计每个标签的数量
    # label_counts = np.bincount(labels, minlength=5)
    #
    # # 打印每个标签的数量
    # for i in range(5):
    #     print(f"Label {i}: {label_counts[i]}")

    # 可视化第一个数据块
    block_data = visualize_block(dataset, block_idx=2)

