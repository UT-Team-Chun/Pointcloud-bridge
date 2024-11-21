import numpy as np
import laspy
import torch
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.cluster import DBSCAN
from tqdm import tqdm


class BridgePointCloudDataset(Dataset):
    def __init__(self,
                 data_dir,
                 horizontal_block_size=20.0,  # 水平方向块大小
                 vertical_block_size=15.0,  # 垂直方向块大小
                 horizontal_stride=16.0,  # 水平方向步长
                 vertical_stride=10.0,  # 垂直方向步长
                 num_points=4096,
                 min_points_in_block=100):  # 每个块的最小点数
        """
        针对桥梁结构的点云数据集
        Args:
            data_dir: 数据目录
            horizontal_block_size: 水平方向块大小（米）
            vertical_block_size: 垂直方向块大小（米）
            horizontal_stride: 水平方向滑动步长（米）
            vertical_stride: 垂直方向滑动步长（米）
            num_points: 每个块采样的点数
            min_points_in_block: 每个块的最小点数阈值
        """
        self.data_dir = Path(data_dir)
        self.h_block_size = horizontal_block_size
        self.v_block_size = vertical_block_size
        self.h_stride = horizontal_stride
        self.v_stride = vertical_stride
        self.num_points = num_points
        self.min_points = min_points_in_block

        self.las_files = list(self.data_dir.glob('*.las'))
        self.blocks = self._preprocess_files()

    def _detect_bridge_direction(self, points):
        """
        检测桥梁的主方向
        使用PCA分析点云的主方向
        """
        # 计算协方差矩阵
        cov_matrix = np.cov(points[:, :2].T)
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # 主方向是最大特征值对应的特征向量
        main_direction = eigenvectors[:, np.argmax(eigenvalues)]
        # 计算旋转角度
        angle = np.arctan2(main_direction[1], main_direction[0])
        return angle

    def _rotate_points(self, points, angle):
        """
        将点云旋转到桥梁的主方向
        """
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        rotated_xy = np.dot(points[:, :2], rotation_matrix.T)
        rotated_points = np.column_stack((rotated_xy, points[:, 2]))
        return rotated_points

    def _detect_bridge_components(self, points, eps=0.5, min_samples=10):
        """
        使用DBSCAN检测桥梁组件
        """
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        return clustering.labels_

    def _preprocess_files(self):
        blocks = []
        for las_file in self.las_files:
            # 读取las文件
            las = laspy.read(las_file)
            points = np.vstack((las.x, las.y, las.z)).T
            colors = np.vstack((las.red, las.green, las.blue)).T / 65535.0
            if hasattr(las, 'classification'):
                labels = las.classification
            else:
                labels = np.zeros(len(points))

            # 检测桥梁主方向并旋转点云
            angle = self._detect_bridge_direction(points)
            rotated_points = self._rotate_points(points, angle)

            # 获取旋转后的点云范围
            min_bounds = rotated_points.min(axis=0)
            max_bounds = rotated_points.max(axis=0)

            # 计算水平和垂直方向的网格数量
            num_blocks_x = int((max_bounds[0] - min_bounds[0] - self.h_block_size) / self.h_stride) + 1
            num_blocks_y = int((max_bounds[1] - min_bounds[1] - self.h_block_size) / self.h_stride) + 1
            num_blocks_z = int((max_bounds[2] - min_bounds[2] - self.v_block_size) / self.v_stride) + 1

            # 分块处理
            for i in tqdm(range(num_blocks_x), desc="Processing blocks"):
                for j in range(num_blocks_y):
                    for k in range(num_blocks_z):
                        # 定义当前块的范围
                        x_min = min_bounds[0] + i * self.h_stride
                        y_min = min_bounds[1] + j * self.h_stride
                        z_min = min_bounds[2] + k * self.v_stride
                        x_max = x_min + self.h_block_size
                        y_max = y_min + self.h_block_size
                        z_max = z_min + self.v_block_size

                        # 选择在当前块内的点
                        mask = (rotated_points[:, 0] >= x_min) & (rotated_points[:, 0] < x_max) & \
                               (rotated_points[:, 1] >= y_min) & (rotated_points[:, 1] < y_max) & \
                               (rotated_points[:, 2] >= z_min) & (rotated_points[:, 2] < z_max)

                        block_points = rotated_points[mask]

                        # 检查点数是否满足最小要求
                        if len(block_points) >= self.min_points:
                            # 检测组件
                            component_labels = self._detect_bridge_components(block_points)

                            blocks.append({
                                'file': las_file,
                                'bounds': (x_min, y_min, z_min, x_max, y_max, z_max),
                                'points': points[mask],  # 原始坐标
                                'rotated_points': block_points,  # 旋转后的坐标
                                'colors': colors[mask],
                                'labels': labels[mask],
                                'component_labels': component_labels,
                                'rotation_angle': angle
                            })

        return blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        block = self.blocks[idx]
        points = block['points']
        rotated_points = block['rotated_points']
        colors = block['colors']
        labels = block['labels']
        component_labels = block['component_labels']

        # 采样处理
        if len(points) > self.num_points:
            choice = np.random.choice(len(points), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(points), self.num_points, replace=True)

        points = points[choice]
        rotated_points = rotated_points[choice]
        colors = colors[choice]
        labels = labels[choice]
        component_labels = component_labels[choice]

        # 标准化
        center = rotated_points.mean(axis=0)
        normalized_points = rotated_points - center

        # 分别对水平和垂直方向进行缩放
        horizontal_scale = np.max(np.abs(normalized_points[:, :2]))
        vertical_scale = np.max(np.abs(normalized_points[:, 2]))

        normalized_points[:, :2] = normalized_points[:, :2] / horizontal_scale
        normalized_points[:, 2] = normalized_points[:, 2] / vertical_scale

        # 转换为tensor
        return {
            'points': torch.FloatTensor(points),
            'rotated_points': torch.FloatTensor(rotated_points),
            'normalized_points': torch.FloatTensor(normalized_points),
            'colors': torch.FloatTensor(colors),
            'labels': torch.LongTensor(labels),
            'component_labels': torch.LongTensor(component_labels),
            'center': torch.FloatTensor(center),
            'scales': torch.FloatTensor([horizontal_scale, vertical_scale]),
            'rotation_angle': torch.FloatTensor([block['rotation_angle']])
        }


if __name__ == '__main__':
    dataset = BridgePointCloudDataset(
        data_dir='../data/val',
        horizontal_block_size=1,  # 桥梁延伸方向使用较大的块
        vertical_block_size=0.5,  # 垂直方向使用较小的块
        horizontal_stride=0.5,  # 水平方向重叠4米
        vertical_stride=0.2,  # 垂直方向重叠5米
        num_points=4096,
        min_points_in_block=100
    )
    # 获取一个数据样本
    sample = dataset[0]
    print(f"Points shape: {sample['points'].shape}")
    print(f"Colors shape: {sample['colors'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")