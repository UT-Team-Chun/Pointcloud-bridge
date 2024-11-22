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
        print(f"Loading {file_path}")
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

            print(f"Loaded {len(points)} points from {filename}")
            self.logger.info(f"Loaded {len(points)} points from {filename}")

            return points, colors, labels

        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            self.logger.error(f"Error loading {filename}: {str(e)}")

            return None, None, None

    def _assign_points_to_blocks(self, points, colors, labels):
        """将点分配到不同的块中"""
        print(f"Processing data - Points: {points.shape}, Colors: {colors.shape}, Labels: {labels.shape}")
        self.logger.info(f"Processing data - Points: {points.shape}, Colors: {colors.shape}, Labels: {labels.shape}")

        # 计算点云的边界
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)

        # 计算全局统计信息
        global_mean = np.mean(points, axis=0)
        global_scale = np.max(np.linalg.norm(points - global_mean, axis=1))

        # 计算每个点所属的网格索引
        grid_indices = np.floor((points - min_coords) / [self.h_stride, self.h_stride, self.v_stride]).astype(int)

        # 使用字典收集每个网格中的点
        grid_dict = {}
        for i, grid_idx in enumerate(grid_indices):
            key = tuple(grid_idx)
            if key not in grid_dict:
                grid_dict[key] = []
            grid_dict[key].append(i)

        blocks = []
        # 处理每个非空的网格
        for grid_idx, indices in grid_dict.items():
            indices = np.array(indices)
            if len(indices) >= self.min_points:
                # 计算块的中心点
                center = min_coords + np.array(grid_idx) * [self.h_stride, self.h_stride, self.v_stride] + \
                         [self.h_block_size / 2, self.h_block_size / 2, self.v_block_size / 2]

                block_points = points[indices]
                block_colors = colors[indices]
                block_labels = labels[indices]

                # 将点坐标归一化到块的中心
                normalized_points = block_points - center
                # 添加全局位置信息
                #block_points = (center - global_mean) / global_scale

                blocks.append({
                    'points': block_points,
                    'colors': block_colors,
                    'labels': block_labels,
                    'center': center,
                    'indices': indices,
                    'normalized_points': normalized_points
                })

        print(f"Created {len(blocks)} blocks")
        self.logger.info(f"Created {len(blocks)} blocks")

        return blocks

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

    def __len__(self):
        """返回数据集中块的数量"""
        return len(self.blocks)

    def __getitem__(self, idx):
        """获取指定索引的数据块"""
        block = self.blocks[idx]

        # 转换为所需的数据格式
        points = self.normalize_points(block['points']).astype(np.float32)
        colors = block['colors'].astype(np.float32)
        labels = block['labels'].astype(np.int64)

        # 统一点数
        if self.num_points is not None:
            if len(points) >= self.num_points:
                # 随机采样到指定点数
                choice = np.random.choice(len(points), self.num_points, replace=False)
            else:
                # 如果点数不足，则重复采样
                choice = np.random.choice(len(points), self.num_points, replace=True)

        points = points[choice]
        colors = colors[choice]
        labels = labels[choice]

        # 应用数据增强
        if self.transform:
            points, colors, labels = self.transform(points, colors, labels)

        return {
            'points': points,
            'colors': colors,
            'labels': labels,
            'center': block['center']
        }

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time


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
        h_block_size=0.5,
        v_block_size=0.5,
        h_stride=0.5,
        v_stride=0.5,
        min_points=100
    )

    # 可视化第一个数据块
    block_data = visualize_block(dataset, block_idx=10)
    # 查看点数分布
    point_counts = []
    for i in range(len(dataset)):
        data = dataset[i]
        point_counts.append(len(data['points']))

    print(f"最小点数: {min(point_counts)}")
    print(f"最大点数: {max(point_counts)}")
    print(f"平均点数: {sum(point_counts) / len(point_counts):.2f}")

    # 打印数据集的总块数
    print(f"\n数据集总块数: {len(dataset)}")
    print(f"数据集总点数: {sum(point_counts)}")
