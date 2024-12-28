import numpy as np
from scipy.spatial import KDTree
from typing import Tuple

def build_graph(
    superpoint_labels: np.ndarray,
    superpoint_features: np.ndarray,
    k_neighbors: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """构建超点图"""
    # 提取超点中心点坐标
    centers = superpoint_features[:, :3]  # 假设前3维是空间坐标
    
    # 构建KD树
    tree = KDTree(centers)
    
    # 找到k近邻
    distances, indices = tree.query(centers, k=k_neighbors + 1)  # +1 因为包含自身
    
    # 构建边
    edge_index = []
    edge_attr = []
    
    for i in range(len(centers)):
        for j, dist in zip(indices[i][1:], distances[i][1:]):  # 跳过第一个（自身）
            # 添加双向边
            edge_index.extend([[i, j], [j, i]])
            
            # 计算边特征
            source_feat = superpoint_features[i]
            target_feat = superpoint_features[j]
            
            # 边特征包括：
            # 1. 距离
            # 2. 特征差异
            # 3. 方向向量
            edge_feat = np.concatenate([
                np.array([dist]),  # 距离
                source_feat - target_feat,  # 特征差异
                centers[j] - centers[i],  # 方向向量
            ])
            
            edge_attr.extend([edge_feat, edge_feat])  # 双向边使用相同特征
    
    return np.array(edge_index).T, np.array(edge_attr)