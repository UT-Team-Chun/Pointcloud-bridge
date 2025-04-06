import logging
import os

import laspy
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BridgePointCloudDataset(Dataset):
    def __init__(self, data_dir, num_points=4096, transform=False):
        """
        Args:
            data_dir (str): 包含.las文件的目录路径
            num_points (int): 采样点数
            transform (bool): 是否进行数据增强
        """
        self.data_dir = data_dir
        self.num_points = num_points
        self.transform = transform

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

        logger.info(f"在 {data_dir} 中找到 {len(self.file_list)} 个.las文件")
        print(f"在 {data_dir} 中找到 {len(self.file_list)} 个.las文件")
        logger.info(f"数据集中的有效标签: {sorted(list(self.valid_labels))}")
        print(f"数据集中的有效标签: {sorted(list(self.valid_labels))}")

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
        return len(self.file_list)

    def __getitem__(self, idx):
        # 加载.las文件
        las_path = self.file_list[idx]
        try:
            las = laspy.read(las_path)

            # 提取点云数据
            points = np.vstack((las.x, las.y, las.z)).T

            # 提取颜色信息（如果有）
            if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                colors = np.vstack((
                    np.array(las.red) / 65535.0,
                    np.array(las.green) / 65535.0,
                    np.array(las.blue) / 65535.0
                )).T
            else:
                colors = np.zeros((points.shape[0], 3))

            # 提取分类标签（如果有）
            if hasattr(las, 'classification'):
                labels = np.array(las.classification)
                # 将超出0-7范围的标签设为0（noise类）
                labels = np.where((labels >= 0) & (labels <= 7), labels, 0)
            else:
                labels = np.zeros(points.shape[0])

            # 随机采样到指定点数
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


    specific_file = ['bridge-7.las']
    # 创建数据集实例
    dataset = BridgePointCloudDataset(
        data_dir='../data/fukushima/onepart/val',  # 替换为你的数据路径
        num_points=4096
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
    block_data = visualize_block(dataset, block_idx=0)