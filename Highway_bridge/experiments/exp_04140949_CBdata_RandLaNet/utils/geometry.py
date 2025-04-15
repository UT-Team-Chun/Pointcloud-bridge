import numpy as np
from typing import Tuple, Optional
import torch

def estimate_local_frame(
    points: np.ndarray,
    k: int = 30
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """估计局部参考系"""
    num_points = points.shape[0]
    eigenvalues = np.zeros((num_points, 3))
    eigenvectors = np.zeros((num_points, 3, 3))
    
    # 构建KD树
    from scipy.spatial import KDTree
    tree = KDTree(points)
    
    for i in range(num_points):
        # 获取近邻点
        distances, indices = tree.query(points[i:i+1], k=k)
        neighbors = points[indices[0]]
        
        # 计算局部协方差矩阵
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered / (k - 1)
        
        # 计算特征值和特征向量
        w, v = np.linalg.eigh(cov)
        
        # 排序（从小到大）
        idx = w.argsort()
        eigenvalues[i] = w[idx]
        eigenvectors[i] = v[:, idx]
    
    return eigenvalues, eigenvectors

def compute_geometric_features(
    points: np.ndarray,
    eigenvalues: np.ndarray
) -> np.ndarray:
    """计算几何特征"""
    # 确保特征值按升序排列
    l1, l2, l3 = eigenvalues[:, 2], eigenvalues[:, 1], eigenvalues[:, 0]
    
    # 计算各种几何特征
    linearity = (l1 - l2) / (l1 + 1e-8)
    planarity = (l2 - l3) / (l1 + 1e-8)
    sphericity = l3 / (l1 + 1e-8)
    
    return np.stack([linearity, planarity, sphericity], axis=1)