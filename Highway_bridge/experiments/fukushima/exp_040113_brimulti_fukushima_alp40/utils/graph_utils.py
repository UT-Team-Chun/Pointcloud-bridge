import torch
import numpy as np
from typing import Tuple, Optional, List
from scipy.spatial import KDTree

def radius_graph(
    pos: np.ndarray,
    r: float,
    max_num_neighbors: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """构建半径图"""
    tree = KDTree(pos)
    
    # 查询半径内的邻居
    pairs = tree.query_pairs(r, output_type='ndarray')
    
    # 转换为双向边
    edges = np.vstack([pairs, pairs[:, ::-1]])
    
    if max_num_neighbors is not None:
        # 限制每个节点的最大邻居数
        unique_nodes, counts = np.unique(edges[:, 0], return_counts=True)
        mask = np.ones(len(edges), dtype=bool)
        
        for node, count in zip(unique_nodes, counts):
            if count > max_num_neighbors:
                node_edges = np.where(edges[:, 0] == node)[0]
                remove_idx = np.random.choice(
                    node_edges,
                    size=count - max_num_neighbors,
                    replace=False
                )
                mask[remove_idx] = False
        
        edges = edges[mask]
    
    # 计算边特征（距离）
    dist = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
    
    return edges, dist

def knn_graph(
    pos: np.ndarray,
    k: int,
    loop: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """构建K近邻图"""
    tree = KDTree(pos)
    distances, indices = tree.query(pos, k=k + 1 if loop else k)
    
    # 构建边
    rows = np.repeat(np.arange(len(pos)), k if not loop else k + 1)
    cols = indices.reshape(-1)
    
    edges = np.stack([rows, cols], axis=1)
    
    if not loop:
        # 移除自环
        mask = edges[:, 0] != edges[:, 1]
        edges = edges[mask]
        distances = distances.reshape(-1)[mask]
    
    return edges, distances