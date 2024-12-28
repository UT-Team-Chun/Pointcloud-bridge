import os
from typing import List

import laspy
import numpy as np
import torch
from torch_geometric.data import Dataset, Data

from .preprocessing.graph_construction import build_graph
from .preprocessing.superpoint_generation import generate_superpoints


class PointCloudDataset(Dataset):
    def __init__(
            self,
            root: str,
            split: str = 'train',
            transform=None,
            pre_transform=None,
            pre_filter=None,
            use_cache: bool = True
    ):
        self.split = split
        self.use_cache = use_cache
        super().__init__(root, transform, pre_transform, pre_filter)

        # 加载文件列表
        self.file_list = self._get_file_list()

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.split)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        """返回原始文件列表"""
        return self.file_list if hasattr(self, 'file_list') else []

    @property
    def processed_file_names(self) -> List[str]:
        """返回处理后的文件列表"""
        return [f'{f}.pt' for f in self.raw_file_names]

    def _get_file_list(self) -> List[str]:
        """获取对应split的文件列表"""
        split_file = os.path.join(self.root, f'{self.split}.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        return []

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        # 获取处理后的文件路径
        processed_path = os.path.join(
            self.processed_dir,
            f'{self.file_list[idx]}.pt'
        )

        # 如果启用缓存且文件存在，直接加载
        if self.use_cache and os.path.exists(processed_path):
            return torch.load(processed_path)

        # 否则进行处理
        data = self._process_raw_data(idx)

        # 如果启用缓存，保存处理后的数据
        if self.use_cache:
            os.makedirs(self.processed_dir, exist_ok=True)
            torch.save(data, processed_path)

        return data

    def _process_raw_data(self, idx) -> Data:
        """处理原始点云数据"""
        # 读取LAS文件
        raw_path = os.path.join(self.raw_dir, self.file_list[idx])
        las = laspy.read(raw_path)

        # 获取点云数据
        points = np.vstack((las.x, las.y, las.z)).T

        # 获取RGB数据并归一化到[0,1]范围
        colors = np.vstack((
            las.red / 65535.0,  # 通常LAS文件中的RGB值范围是0-65535
            las.green / 65535.0,
            las.blue / 65535.0
        )).T

        # 获取标签
        labels = np.array(las.classification)

        # 计算法向量（如果需要的话）
        # 这里我们可以使用PCA方法或者其他方法来估计法向量
        normals = self._estimate_normals(points)

        # 生成超点
        superpoints, sp_features = generate_superpoints(
            points=points,
            colors=colors,
            normals=normals,
            labels=labels  # 传入标签信息
        )

        # 构建图
        edge_index, edge_attr = build_graph(superpoints, sp_features)

        # 创建Data对象
        data = Data(
            x=torch.FloatTensor(sp_features),
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_attr),
            pos=torch.FloatTensor(points),
            y=torch.LongTensor(labels),  # 添加标签
            superpoint_labels=torch.LongTensor(superpoints)
        )

        return data

    def _estimate_normals(self, points: np.ndarray, k: int = 30) -> np.ndarray:
        """估计点云法向量"""
        from sklearn.neighbors import NearestNeighbors

        # 初始化法向量数组
        normals = np.zeros_like(points)

        # 构建KNN
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
        distances, indices = nbrs.kneighbors(points)

        # 对每个点计算法向量
        for i in range(len(points)):
            # 获取近邻点
            neighbors = points[indices[i]]

            # 计算协方差矩阵
            centered = neighbors - neighbors.mean(axis=0)
            cov = centered.T @ centered

            # 计算特征值和特征向量
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # 最小特征值对应的特征向量即为法向量
            normal = eigenvectors[:, 0]

            # 确保法向量指向外部（这是一个启发式方法，可能需要根据具体数据调整）
            if normal.dot(points[i] - points.mean(axis=0)) < 0:
                normal = -normal

            normals[i] = normal

        return normals


# 使用示例
if __name__ == '__main__':
    dataset = PointCloudDataset(
        root='path/to/your/data',
        split='train'
    )

    # 测试数据加载
    data = dataset[0]
    print(f'Number of points: {data.pos.shape[0]}')
    print(f'Number of features: {data.x.shape[1]}')
    print(f'Number of classes: {len(torch.unique(data.y))}')