import numpy as np
from sklearn.cluster import DBSCAN
from typing import Tuple, List
import torch
from scipy.spatial import KDTree

def compute_geometric_features(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """计算几何特征"""
    # 计算局部曲率
    tree = KDTree(points)
    k = 30  # k近邻数量
    geometric_features = []
    
    for i in range(len(points)):
        # 找到近邻点
        distances, indices = tree.query(points[i:i+1], k=k)
        neighbor_points = points[indices[0]]
        neighbor_normals = normals[indices[0]]
        
        # 计算局部协方差矩阵
        centered_points = neighbor_points - neighbor_points.mean(axis=0)
        cov = centered_points.T @ centered_points / k
        
        # 计算特征值
        eigenvalues = np.linalg.eigvals(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # 计算几何特征
        linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
        planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]
        sphericity = eigenvalues[2] / eigenvalues[0]
        
        geometric_features.append([linearity, planarity, sphericity])
    
    return np.array(geometric_features)

def generate_superpoints(
    points: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    min_points: int = 20,
    eps: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """生成超点和其特征"""
    # 计算几何特征
    geometric_features = compute_geometric_features(points, normals)
    
    # 组合特征用于聚类
    features_for_clustering = np.concatenate([
        points,  # 空间位置
        normals * 0.5,  # 法向量（权重较小）
        geometric_features * 2.0,  # 几何特征（权重较大）
        colors * 0.3  # 颜色特征（权重适中）
    ], axis=1)
    
    # 使用DBSCAN进行聚类
    clustering = DBSCAN(
        eps=eps,
        min_samples=min_points,
        metric='euclidean',
        n_jobs=-1
    ).fit(features_for_clustering)
    
    superpoint_labels = clustering.labels_
    
    # 计算每个超点的特征
    unique_labels = np.unique(superpoint_labels)
    superpoint_features = []
    
    for label in unique_labels:
        if label == -1:  # 跳过噪声点
            continue
            
        mask = superpoint_labels == label
        sp_points = points[mask]
        sp_colors = colors[mask]
        sp_normals = normals[mask]
        sp_geometric = geometric_features[mask]
        
        # 计算特征
        feature = np.concatenate([
            sp_points.mean(axis=0),  # 中心点
            sp_colors.mean(axis=0),  # 平均颜色
            sp_normals.mean(axis=0),  # 平均法向量
            sp_geometric.mean(axis=0),  # 平均几何特征
            sp_points.std(axis=0),  # 空间分布
            np.array([len(sp_points)])  # 点数量
        ])
        
        superpoint_features.append(feature)
    
    return superpoint_labels, np.array(superpoint_features)